"""Gaussian-process MPC controller for model-error compensation."""

from smallsat_sim.controllers.base_mpc_controller import BaseMPCController
from smallsat_sim.envs.base_env import BaseEnv

# Utils
from smallsat_sim.utils.helpers import (
    calc_model_error,
    calc_attitude_error,
    calc_lateral_tracking_error,
)
from smallsat_sim.utils.logger import Logger
from smallsat_sim.controllers.gp_mpc.online_learning.utils import (
    ScaleFeatureSelector,
    ResidualScaler,
)

# General libraries
import gpytorch
from gpytorch.constraints.constraints import Positive
import os
import tempfile
from smallsat_sim.controllers.state import body_state
import numpy as np
import torch
import casadi as ca
from scipy.stats import norm
import time

# Acados
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    ZoroDescription,
)
from casadi import SX

# Zero Order GPMPC
# import zero_order_gpmpc
# from zero_order_gpmpc.controllers import (
#     ZeroOrderGPMPC,
# )
# from zero_order_gpmpc.controllers.zoro_acados_utils import (
#     setup_sim_from_ocp,
# )
from l4acados.controllers.zero_order_gpmpc import ZeroOrderGPMPC
from l4acados.controllers.zoro_acados_utils import setup_sim_from_ocp

# gpytorch utilities
# from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_hyperparam_training import (
#     generate_train_inputs_acados,
#     generate_train_outputs_at_inputs,
#     train_gp_model,
# )
# from l4acados.external.gpytorch_utils.gp_hyperparam_training import generate_train_inputs_acados, generate_train_outputs_at_inputs, train_gp_model
# from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_utils import (
#     gp_data_from_model_and_path,
#     gp_derivative_data_from_model_and_path,
#     plot_gp_data,
#     generate_grid_points,
# )




# from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
#     GPyTorchResidualModel,
# )
# from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_learning_model import (
#     GPyTorchResidualLearningModel,
# )

from l4acados.models import GPyTorchResidualModel
# from l4acados.models.pytorch_models.gpytorch_models import GPyTorchResidualLearningModel # no longer exists?!!!!


# GPyTorch models
# from zero_order_gpmpc.models.gpytorch_models.gpytorch_gp import (
#     BatchIndependentMultitaskGPModel,
# )
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import BatchIndependentMultitaskGPModel

# Import DataProcessing strategies from SmallSatSim
from smallsat_sim.controllers.gp_mpc.online_learning.strategies import (
    SlidingWindow,
    SlidingWindowPlus,
)


# Set default torch dtype
torch.set_default_dtype(torch.float64)


class GPMPC(BaseMPCController):
    """
    This class implements a GP MPC controller based on GPyTorch and acados.

    See the following documentations for details:
    GPyTorch: https://docs.gpytorch.ai/en/stable/
    acados: https://docs.acados.org/
    """

    def __init__(self, env: BaseEnv, planner) -> None:
        # Fetch correct controller config
        from smallsat_sim.planners.validation import require_contouring_planner
        require_contouring_planner(planner)
        self.ctrl_cfg = env.env_cfg.control.GPMPC
        self.N = self.ctrl_cfg.N

        # Initialize base class
        super().__init__(env, planner, self.ctrl_cfg)

        # Save relevant components from environment locally
        self.f = env.symbolic_model.f_expl_expr_func
        # Training targets must use the same integration scheme as prediction;
        # otherwise the GP learns Euler discretization error even without a fault.
        from smallsat_sim.controllers.nominal_mpc.reference import rk4_step
        model = env.symbolic_model
        integrator = ca.Function("gp_nominal_step", [model.x, model.u],
                                 [rk4_step(self.f, model.x, model.u, self.ctrl_cfg.Ts)])
        self.f_int = lambda x, u: integrator(x, u).toarray()

        # # Initialize logger
        # self.logger = Logger()
        # _, self.logged_data = self.logger.load_data_from_hdf5(
        #     "mujoco_log_20240707_145725.h5"
        # )

        # Generate solver
        self._generate_nominal_ocp(env)

        # Create Zoro description
        self._create_zoro_description(env)

        # Setup Gaussian Process
        self._setup_gp(obs=body_state(env))

        # Miscellaneous
        self.last_solution = {
            "states": np.tile(
                np.zeros(
                    self.nx,
                ),
                (self.N + 1, 1),
            ),
            "inputs": np.tile(
                np.zeros(
                    self.nu,
                ),
                (self.N, 1),
            ),
        }

        # Initialize solver
        self._initialize_solver(env)

        self.visualization = getattr(env, "visualization", None)
        self._record_video = bool(getattr(getattr(env, "args", None), "video", False))
        from copy import deepcopy
        self._fresh_residual_model = deepcopy(self.gp_mpc.residual_model)
        self._previous_time = None

    def reset(self, env):
        """Clear trial-specific training data, residual history and warm starts."""
        from copy import deepcopy
        self.gp_mpc.residual_model = deepcopy(self._fresh_residual_model)
        self._previous_time = None
        self.x_past = np.r_[body_state(env), 0.]
        self.u_past = np.zeros(self.nu)
        self.gp_mpc.ocp_solver.reset()
        self._initialize_solver(env)

    def _setup_gp(self, obs: np.ndarray) -> None:
        """
        Initializes all needed quantities for the Gaussian Process
        """
        # # Initialize empty feature tensor (z)
        # self.z = torch.from_numpy(self.logged_data["z"].squeeze(1))

        # # Initialize empty ouput tensor (y)
        # self.y = torch.from_numpy(self.logged_data["y"].squeeze(1))

        # Initialize subspace matrix B_d
        # Shape: (nx, 6)
        self.B_d = torch.cat((torch.zeros(7, 6), torch.eye(6), torch.zeros(1, 6))).to(
            torch.float64
        )
        self.B_d_inv = torch.linalg.pinv(self.B_d).to(torch.float64)

        # Initialize buffers to keep track of past (phyisical) states and inputs
        self.x_past = np.zeros(self.nx)
        self.x_past[:-1] = obs
        self.u_past = np.zeros(self.nu)

        # Initialize some hyperparameters for GP
        # TODO: Move to configuration file
        self.M = int(getattr(self.ctrl_cfg, "max_points", 300))
        self.gp_update_counter = 0  # Keep track how many times dict has been updated
        self.gp_initialized = False  # Keep track if GP is already initialized

        # Setup residual model with trained GP
        input_feature_selection = np.r_[np.zeros(self.nx, dtype=int),
                                        np.ones(self.nu - 1, dtype=int), 0]
        self.input_selection = ScaleFeatureSelector(input_feature_selection, None)
        self.residual_scaler = ResidualScaler(scale=1)

        # Initialize likelihood and overwrite default values
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=6, noise_constraint=Positive()
        )

        # Hardcode noise parameters in likelihood
        likelihood.noise = torch.tensor([1e-10])
        likelihood.raw_task_noises.data = torch.tensor(
            [-30.0, -30.0, -30.0, -30.0, -30.0, -30.0]
        )

        # Initialize GP model and overwrite default values
        mode = "Nonlinear Kernel"
        gp_model = BatchIndependentMultitaskGPModel(
            train_x=None,
            train_y=None,
            likelihood=likelihood,
            use_ard=True,
            residual_dimension=6,
            input_dimension=sum(input_feature_selection),
        )

        for name, param in gp_model.named_parameters():
            print(f"Parameter {name} has shape {param.shape} and values:")
            print(param)

        if hasattr(self.ctrl_cfg, "prior_variance"):
            gp_model.covar_module.outputscale = float(self.ctrl_cfg.prior_variance)
        if hasattr(self.ctrl_cfg, "lengthscale"):
            gp_model.covar_module.base_kernel.lengthscale = float(self.ctrl_cfg.lengthscale)
        if hasattr(self.ctrl_cfg, "observation_noise"):
            likelihood.noise = torch.tensor([float(self.ctrl_cfg.observation_noise)])

        # Set hyperparemters of linear kernel manually.
        gp_model = self._initialize_hyperparameters(gp_model, likelihood, mode)

        # Initialize the Residual Model
        # residual_model = GPyTorchResidualLearningModel(
        #     gp_model=gp_model,
        #     gp_feature_selector=self.input_selection,
        #     residual_scaler=self.residual_scaler,
        #     data_processing_strategy=SlidingWindowPlus(
        #         max_num_points=self.M, device=next(gp_model.parameters()).device.type
        #     ),
        #     verbose=False,
        # )
        residual_model = GPyTorchResidualModel(
            gp_model=gp_model,
            feature_selector=self.input_selection,
            # gp_feature_selector=self.input_selection,
            # residual_scaler=self.residual_scaler,
            data_processing_strategy=SlidingWindowPlus(
                max_num_points=self.M, device=next(gp_model.parameters()).device.type
            ),
            # verbose=False,
        )

        # File naming stuff
        json_ocp = "zoro_ocp_solver_config.json"
        json_sim = "zoro_sim_solver_config.json"
        filename_ocp = os.path.join(self.save_dir, json_ocp)
        filename_sim = os.path.join(self.save_dir, json_sim)

        self.ocp_init.code_export_directory = os.path.join(
            self.save_dir, "c_generated_code_ocp"
        )

        self.nominal_sim.code_export_directory = os.path.join(
            self.save_dir, "c_generated_code_sim"
        )

        # Generate ZeroOrderGPMPC
        self.gp_mpc = ZeroOrderGPMPC(
            self.ocp_init,
            residual_model=residual_model,
            path_json_ocp=filename_ocp,
            path_json_sim=filename_sim,
            build_c_code=True,
            use_cython=False,  # TODO: Check why not supported
            B=self.B_d.numpy(),
        )

    def _initialize_hyperparameters(self, gp_model, likelihood, mode):
        if mode == "Linear Kernel":
            gp_model.covar_module.variance = torch.tensor([1e-2])
        elif mode == "Nonlinear Kernel":
            pass
        else:
            raise RuntimeError(f"Mode {mode} not known.")

        gp_model.eval()
        likelihood.eval()

        return gp_model

    def _generate_nominal_ocp(self, env) -> None:
        """
        Generates the nominal ocp (w/o residual dynamics)
        """
        # Initialize OCP instance
        ocp = AcadosOcp()

        # Fetch symbolic model
        model = env.symbolic_model

        # Setup acados model (interface between acados and CasADi)
        acados_model = AcadosModel()
        acados_model.f_impl_expr = model.f_impl_expr
        acados_model.f_expl_expr = model.f_expl_expr
        acados_model.x = model.x
        acados_model.xdot = model.xdot
        acados_model.u = model.u
        acados_model.z = model.z
        acados_model.p = model.p
        acados_model.name = "OCPsolver"

        # Define theta and dtheta
        theta = ca.SX.sym("theta", 1, 1)
        d_theta = ca.SX.sym("d_theta", 1, 1)

        acados_model.x = ca.vertcat(acados_model.x, theta)
        acados_model.u = ca.vertcat(acados_model.u, d_theta)
        theta_dot = ca.SX.sym("theta_dot")
        acados_model.xdot = ca.vertcat(acados_model.xdot, theta_dot)

        acados_model.f_expl_expr = ca.vertcat(acados_model.f_expl_expr, d_theta)

        acados_model.f_impl_expr = ca.vertcat(acados_model.f_impl_expr, theta_dot - d_theta)

        # Assign parameters and model
        Ts = self.ctrl_cfg.Ts
        p_start = SX.sym("p_start", (3, 1))  # Start point of line segment<
        t = SX.sym("t", (3, 1))  # Direction of line segment
        theta_start = SX.sym("theta_start")  # Arclength start of line segment
        q_des = SX.sym("q_des", (4, 1))  # Desired attitude
        q_theta = SX.sym("q_theta", (1,1))

        p = ca.vertcat(p_start, t, theta_start, q_des, q_theta)
        acados_model.p = p
        ocp.model = acados_model

        # Define and assign cost functions
        R = self.ctrl_cfg.cost.R
        q_l = self.ctrl_cfg.cost.q_l
        Q_c = self.ctrl_cfg.cost.Q_c
        Q_omega = self.ctrl_cfg.cost.Q_omega
        #q_theta = self.ctrl_cfg.cost.q_theta
        Q_q = self.ctrl_cfg.cost.Q_q

        # Calculate line representation
        tx, ty, tz = t[0], t[1], t[2]
        g = p_start + (theta - theta_start) * t

        # Extract states for ease of use
        r = model.x[0:3]
        q = model.x[3:7]
        v = model.x[7:10]
        omega = model.x[10:13]

        # Calculate errors
        e = r - g
        e_l = t.T @ e

        # Normal projection matrix
        P_n = ca.SX(3, 3)
        P_n[0, 0] = 1 - tx**2
        P_n[0, 1] = -tx * ty
        P_n[0, 2] = -tx * tz
        P_n[1, 0] = -tx * ty
        P_n[1, 1] = 1 - ty**2
        P_n[1, 2] = -ty * tz
        P_n[2, 0] = -tx * tz
        P_n[2, 1] = -ty * tz
        P_n[2, 2] = 1 - tz**2

        # Contouring error
        e_c = P_n @ e

        from smallsat_sim.controllers.nominal_mpc.reference import attitude_cost
        rotation_cost = attitude_cost(q, q_des, Q_q)

        # Setup cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (
            q_l * e_l * e_l
            + e_c.T @ Q_c @ e_c
            + rotation_cost
            + omega.T @ Q_omega @ omega
            + (model.u.T) @ R @ (model.u)
            - q_theta * d_theta
        )

        # Create a casadi cost function for numerical evaluation
        self.cost_function = ca.Function(
            "cost_function",
            [acados_model.x, acados_model.u, p_start, t, theta_start, q_des, q_theta],
            [ocp.model.cost_expr_ext_cost],
        )

        # Nonlinear constraint
        acados_model.con_h_expr = ca.vertcat(e.T @ e,
                                             v.T @ v
        )
        ocp.constraints.lh = np.array([0.0, 0.0])
        ocp.constraints.uh = np.array([1.0 * 1.0, 0.25 * 0.25])

        # Terminal constraint
        acados_model.con_h_expr_e = acados_model.con_h_expr
        ocp.constraints.lh_e = np.array([0.0, 0.0])
        ocp.constraints.uh_e = np.array([1.0 * 1.0, 0.25 * 0.25])

        # Set OCP dimensions
        nx = acados_model.x.size()[0]  # number of states
        nu = acados_model.u.size()[0]  # number of inputs
        ocp.dims.nx = nx
        self.nx = nx
        ocp.dims.nsbx = nx
        ocp.dims.nu = nu
        self.nu = nu
        ocp.dims.np = p.size()[0]  # number of parameters
        ocp.dims.N = self.ctrl_cfg.N  # prediction horizon length

        # Nonlinear constraints
        if acados_model.con_h_expr is not None:
            ocp.dims.nh = acados_model.con_h_expr.size()[0]
            ocp.dims.nsh = 1
        else:
            ocp.dims.nh = 0
            ocp.dims.nsh = 0
        ocp.constraints.idxsh = np.array(range(ocp.dims.nsh))

        # Terminal nonlinear constraints
        if acados_model.con_h_expr_e is not None:
            ocp.dims.nh_e = acados_model.con_h_expr_e.size()[0]
            ocp.dims.nsh_e = 1
        else:
            ocp.dims.nh_e = 0
            ocp.dims.nsh_e = 0

        # Total number of slacks at stages (1, N-1)
        ocp.dims.ns = nx + ocp.dims.nsh

        # Define state constraints
        # Lower bound constraints for intermediate stages
        ocp.constraints.lbx = np.array(
            [
                -100,  # x
                -100,  # y
                -100,  # z
                -1.0,  # q[0]
                -1.0,  # q[1]
                -1.0,  # q[2]
                -1.0,  # q[3]
                -0.25,  # vx
                -0.25,  # vy
                -0.25,  # vz
                -0.1,  # omega_x
                -0.1,  # omega_y
                -0.1,  # omega_z
                0,  # theta
            ]
        )

        # Upper bound constraints for intermediate stages
        ocp.constraints.ubx = np.array(
            [
                100,  # x
                100,  # y
                100,  # z
                1.0,  # q[0]
                1.0,  # q[1]
                1.0,  # q[2]
                1.0,  # q[3]
                0.25,  # vx
                0.25,  # vy
                0.25,  # vz
                0.1,  # omega_x
                0.1,  # omega_y
                0.1,  # omega_z
                1000,  # theta
            ]
        )

        # Indexes of the state variables to which the constraints apply
        ocp.constraints.idxbx = np.arange(nx)

        # Slacks on lower/upper bounds
        ocp.constraints.lsbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.usbx = np.zeros(ocp.dims.nsbx)
        ocp.constraints.idxsbx = np.arange(nx)

        ocp.cost.Zl = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.Zu = 5e02 * np.ones(ocp.dims.ns)
        ocp.cost.zl = 5e03 * np.ones(ocp.dims.ns)
        ocp.cost.zu = 5e03 * np.ones(ocp.dims.ns)

        # Define input constraints
        # Fetch thurster limits from the model configuration
        thruster_forces = [
            thruster.forcerange for thruster in env.model_cfg.actuators
        ]
        ocp.constraints.lbu = np.array([forces[0] for forces in thruster_forces])
        ocp.constraints.ubu = np.array([forces[1] for forces in thruster_forces])

        # Attach dtheta constraints
        ocp.constraints.lbu = np.append(ocp.constraints.lbu, 0.0)
        ocp.constraints.ubu = np.append(ocp.constraints.ubu, getattr(self.ctrl_cfg, "progress_rate_limit", 0.2))
        ocp.constraints.idxbu = np.arange(nu)

        # Set intial condition
        ocp.constraints.idxbx_0 = np.arange(nx)
        ocp.constraints.lbx_0 = ocp.constraints.lbx
        ocp.constraints.ubx_0 = ocp.constraints.ubx
        ocp.parameter_values = np.zeros(ocp.dims.np)

        # Configure solver options
        ocp.solver_options.Tsim = Ts
        ocp.solver_options.tf = Ts * self.ctrl_cfg.N
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # l4acados retains this value for its Python outer loop even though
        # the generated SQP_RTI solver itself performs only one iteration.
        ocp.solver_options.nlp_solver_max_iter = int(getattr(self.ctrl_cfg, "nlp_iterations", 1))
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.regularize_method = "PROJECT"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0

        # Set code generation directory
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = getattr(self.ctrl_cfg, "code_export_directory", None) or tempfile.mkdtemp(prefix="smallsat_gpmpc_")
        ocp.code_export_directory = self.save_dir

        # Save ocp for further use
        self.ocp_init = ocp

        # Create integrator for nominal model
        self.nominal_sim = setup_sim_from_ocp(self.ocp_init)

    def _create_zoro_description(self, env: BaseEnv) -> None:
        """
        Creates a zoro description for the GP interface
        """
        # Uncertainty description
        Sigma_x0 = np.diag(
            [
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
                1.0e-6,
            ]
        )

        Sigma_W = np.diag(
            [
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
            ]
        )

        unc_jac_G_mat = np.diag(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ]
        )

        unc_jac_G_mat = unc_jac_G_mat[:, ~np.all(unc_jac_G_mat == 0, axis=0)]

        # create zoro_description
        zoro_description = ZoroDescription()
        zoro_description.unc_jac_G_mat = unc_jac_G_mat
        zoro_description.backoff_scaling_gamma = (
            1  # constraint tighenting (by how many sigma)
        )
        zoro_description.P0_mat = Sigma_x0  # uncertainty on initial state
        zoro_description.fdbk_K_mat = np.zeros(
            (self.ocp_init.dims.nu, self.ocp_init.dims.nx)
        )
        # zoro_description.unc_jac_G_mat = B
        """G in (nx, nw) describes how noise affects dynamics. I.e. x+ = ... + G@w"""
        zoro_description.W_mat = Sigma_W  # covariance of noise entering the system
        """W in (nw, nw) describes the covariance of the noise on the system"""
        zoro_description.input_P0_diag = True
        zoro_description.input_P0 = False
        zoro_description.input_W_diag = True
        zoro_description.input_W_add_diag = True
        zoro_description.output_P_matrices = True
        zoro_description.idx_uh_t = [0]
        self.ocp_init.zoro_description = zoro_description

    def get_control_input(self, env) -> np.ndarray:
        """
        Calculate the control input based on current observation
        """
        # Retrieve some data which are used in multiple functions
        obs = body_state(env)
        self.run_id = env.run_id
        self.timestamp = env.data.time
        if self._previous_time is not None and self.timestamp < self._previous_time:
            self.reset(env)

        # Check if new observation shall be added to dictionary
        res_output = (self._compute_residual(obs) if getattr(self.ctrl_cfg, "learning_enabled", True)
                      and self._previous_time is not None
                      and np.isclose(self.timestamp - self._previous_time, self.ctrl_cfg.Ts)
                      else None)

        # If logging enabled, calculate the prediction errors
        # NOTE: Must be done here, before new observation is added
        # to the GP's dictionary
        if res_output is not None:
            self._calc_prediction_errors(env)

        if res_output is not None:
            residual, x_train = res_output
            residual = self.residual_scaler(residual)
      
            start_time = time.perf_counter()
            self.gp_mpc.residual_model.record_datapoint(
                x_input=x_train, y_target=residual, timestamp=env.data.time
            )
            end_time = time.perf_counter()
            if self.has_logger:
                self.logger.log(run_id=self.run_id, timestamp=self.timestamp, record_time=((end_time-start_time)*1000))

        # print(f"Total Update GP time: {(end_time-start_time)}")

        # Check solver status and re-initialize if needed
        if self.gp_mpc.ocp_solver.status != 0:
            print(f"Solution suboptimal, solver status: {self.gp_mpc.ocp_solver.status}")
            self._initialize_solver(env)

        # Set parameters
        self._set_params()

        # Warm start  solver
        for i in range(self.ctrl_cfg.N):
            i_next = i
            i_next = min(i_next + 1, self.N - 1)
            self.gp_mpc.ocp_solver.set(i, "x", self.last_solution["states"][i_next])
            self.gp_mpc.ocp_solver.set(i, "u", self.last_solution["inputs"][i_next])

        # Set initial condition
        _, theta_init = self.planner.closest_point_on_trajectory(obs[0:3])
        # Geometric projection wraps every lap, whereas the warm start and
        # segment origins use unwrapped arc length. Keep the same lap branch.
        from smallsat_sim.controllers.nominal_mpc.reference import unwrap_progress
        theta_init = unwrap_progress(theta_init, self.theta_prev[1], self.planner.trajectory.length)
        xinit = np.append(obs, theta_init)
        self.gp_mpc.ocp_solver.set(0, "lbx", xinit)
        self.gp_mpc.ocp_solver.set(0, "ubx", xinit)

        # Solve for the first control input in receding horizon fashion
        start_time = time.perf_counter()
        self._previous_time = None
        try:
            status = self.gp_mpc.solve()
        except Exception as error:
            status = int(self.gp_mpc.ocp_solver.status)
            if status != 0:
                from smallsat_sim.controllers.nominal_mpc.reference import MPCSolverError
                raise MPCSolverError(status) from error
            raise
        end_time = time.perf_counter()

        if self.has_logger:
            self.logger.log(run_id=self.run_id, timestamp=self.timestamp, solve_time=((end_time-start_time)*1000))

        from smallsat_sim.controllers.nominal_mpc.reference import require_valid_control
        require_valid_control(status, np.zeros(1))
        self.X_res, self.U_res = self.gp_mpc.get_solution()
        require_valid_control(status, self.X_res)
        require_valid_control(status, self.U_res)
        u0 = self.U_res[0, :]
        require_valid_control(status, u0, self.ocp_init.constraints.lbu, self.ocp_init.constraints.ubu)

        # Visualize
        self._visualize_prediction()

        if self._record_video:
            _, _ = self.planner.get_reference(env.obs)

        # Save current solution
        for i in range(self.N):
            self.last_solution["states"][i] = self.gp_mpc.ocp_solver.get(i, "x")
            self.last_solution["inputs"][i] = self.gp_mpc.ocp_solver.get(i, "u")
        self.last_solution["states"][self.N] = self.gp_mpc.ocp_solver.get(self.N, "x")

        # Save current observation and input
        self.x_past[:-1], self.u_past = obs, u0.copy()
        self._previous_time = self.timestamp

        # Save theta for next iteration
        for i in range(self.ctrl_cfg.N + 1):
            self.theta_prev[i] = self.gp_mpc.ocp_solver.get(i, "x")[-1]

        # Log quantities
        self._log(run_id=env.run_id, timestamp=env.data.time, env=env)

        # Return output
        # NOTE: Very important to copy input to not mess with reference stuff
        return u0[:-1].copy()

    def _set_params(self) -> None:
        """
        Sets the parameters of the solver at runtime
        """
        # Shift the previous solution for theta by one for reinitialization
        theta_shifted = self.theta_prev.copy()
        theta_shifted.append(theta_shifted[-1])
        theta_shifted.pop(0)

        # Set parameters
        for i in range(self.ctrl_cfg.N + 1):

            theta_curr = theta_shifted[i]

            p_start = self.planner.trajectory._get_start_point_segment(theta_curr)
            t = self.planner.trajectory._get_tangent_segment(theta_curr)
            theta_1 = self.planner.trajectory._get_start_arc_length_segment(theta_curr)
            q_des = self.planner.trajectory.get_intermediate_reference(
                theta_curr
            ).attitude
            q_theta = self.compute_q_theta(
                distance=self.planner.distance_to_closest_waypoint(
                    self.planner.trajectory.get_intermediate_reference(
                        theta_curr
                    ).position
                ),i=i
            )

            ref = np.concatenate((p_start, t, theta_1, q_des, np.array([q_theta])))

            self.gp_mpc.p_hat_nonlin[i, :] = ref.flatten()

    def compute_q_theta(self, distance, i):
        """
        Computes q_theta based on the distance to the waypoint.
        q_theta transitions smoothly from 5e-2 to 5e-3 as distance decreases from 1 meter to 0.
        """
        if hasattr(self.ctrl_cfg, "progress_weight"):
            return float(self.ctrl_cfg.progress_weight)
        upper = 5e-2
        lower = 5e-3
        if distance >= 1.5:
            return upper  # 0.05
        elif distance <= 0.0:
            return lower  # 0.005
        else:
            if i == 0:
                #print(f"Distance {distance}")
                pass
            t = distance / 1.5  # Normalize distance to [0, 1]
            q_theta = lower + (upper - lower) * (3 * t**2 - 2 * t**3)
            return q_theta

    def _compute_residual(self, x_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes residual between actual state and expected state

        y_k = B_d^(-1) (x_{k+1} - f(x_k,u_k))
        """

        # Calculate the prediction/model error
        model_error = calc_model_error(
            obs=x_next,
            x_past=self.x_past[:-1],
            u_past=self.u_past[:-1],
            f_int=self.f_int,
        )

        # Ignore last row as theta not relevant
        residual = (self.B_d_inv[:, :-1] @ model_error).squeeze(-1)

        x_train = np.hstack((self.x_past, self.u_past))

        if self.has_logger:
            # Calculate the predicted residual
            with torch.no_grad():
                x_star = torch.atleast_2d(
                    self.input_selection(torch.from_numpy(x_train))
                )
                observed_pred = self.gp_mpc.residual_model.gp_model.likelihood(
                    self.gp_mpc.residual_model.gp_model(x_star))

            # Get mean and confidence intervals
            mean = observed_pred.mean
            stddev = observed_pred.stddev

            # Make sure it's numerically stable
            stddev[stddev < 1e-5] = torch.max(stddev.max().clone().detach(), torch.tensor(1e-5, dtype=stddev.dtype))

            lower = mean - 2*stddev
            upper = mean + 2*stddev

            try:
                n_points_GP = self.gp_mpc.residual_model.gp_model.train_inputs[0].shape[
                    -2
                ]
            except:
                n_points_GP = 0

            # Log quantities
            self.logger.log(
                run_id=self.run_id,
                timestamp=self.timestamp,
                gp_GT=residual.squeeze(0).cpu().numpy(),
                gp_pred=mean.squeeze(0).cpu().numpy(),
                gp_lower=lower.squeeze(0).cpu().numpy(),
                gp_upper=upper.squeeze(0).cpu().numpy(),
                n_points_GP=n_points_GP,
            )

            # Check if we want to log training data too
            if np.abs(self.timestamp % 10) < 0.1 and self.timestamp > 0.0:
                self.logger.log(
                    run_id=self.run_id,
                    timestamp=self.timestamp,
                    x_train=self.gp_mpc.residual_model.gp_model.train_inputs[0],
                    y_train=self.gp_mpc.residual_model.gp_model.train_targets,
                )

        return residual, x_train

    def _calc_prediction_errors(self, env: BaseEnv) -> None:
        """
        Calculates the prediction errors of both:
            - Nominal model
            - Nominal model + learned residual
        """
        if (
            self.has_logger
            and self.gp_mpc.residual_model.gp_model.train_inputs is not None
        ):
            # Retrieve obs (NOTE: Use GT obs here?)
            obs = body_state(env)

            # Calculate nominal prediction error
            e_nom = calc_model_error(
                obs=obs,
                x_past=self.x_past[:-1],
                u_past=self.u_past[:-1],
                f_int=self.f_int,
            )

            # Calculate GP prediction error
            x_test = torch.from_numpy(np.hstack((self.x_past, self.u_past))).unsqueeze(
                0
            )
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                z_test = self.input_selection(x_test)
                predicted_residual = self.gp_mpc.residual_model.gp_model(z_test)
                predicted_residual = self.residual_scaler(predicted_residual)
            e_gp = e_nom - torch.matmul(self.B_d[:-1, :], predicted_residual.mean.T)

            # L2 norm of both
            e_gp = torch.norm(e_gp, 2).item()
            e_nom = torch.norm(e_nom, 2).item()

            # Log quantities
            if self.has_logger:
                self.logger.log(
                    run_id=env.run_id, timestamp=env.data.time, e_gp=e_gp, e_nom=e_nom
                )

    def _initialize_solver(self, env: BaseEnv) -> None:
        """
        Initializes the solver. Also known as "warm start".
        """
        # Retrieve closest point on track (relevant for theta)
        _, theta_init = self.planner.closest_point_on_trajectory(body_state(env)[0:3])

        # Array to store previous theta
        self.theta_prev = [theta_init for i in range(self.ctrl_cfg.N + 1)]

        x_guess = np.r_[body_state(env), theta_init]
        for i in range(self.N + 1):
            self.gp_mpc.ocp_solver.set(i, "x", x_guess)
            self.last_solution["states"][i] = x_guess
            if i < self.N:
                command = np.zeros(self.nu)
                self.gp_mpc.ocp_solver.set(i, "u", command)
                self.last_solution["inputs"][i] = command

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            obs_gt = (
                body_state(env)
            )  # TODO: Change this to get GT obs, once MR has been merged

            # Tracking error
            tracking_error = calc_lateral_tracking_error(
                obs=obs_gt, planner=self.planner
            )

            # Attitude error
            _, curr_arc_length = self.planner.closest_point_on_trajectory(
                point=obs_gt[:3]
            )
            q_ref = self.planner.trajectory.get_intermediate_reference(
                curr_arc_length
            ).attitude

            attitude_error = calc_attitude_error(q_ref=q_ref, q=obs_gt[3:7])

            # Track cost value of current solution
            mpc_cost = self.cost_function(
                self.gp_mpc.ocp_solver.get(0, "x"),
                self.gp_mpc.ocp_solver.get(0, "u"),
                self.gp_mpc.p_hat_nonlin[0, 0:3].copy(),
                self.gp_mpc.p_hat_nonlin[0, 3:6].copy(),
                self.gp_mpc.p_hat_nonlin[0, 6].copy(),
                self.gp_mpc.p_hat_nonlin[0, 7:11].copy(),
                self.gp_mpc.p_hat_nonlin[0, 11].copy()
            ).full()

            velocity = np.linalg.norm(obs_gt[7:10])

            # Log quantities
            self.logger.log(
                run_id=run_id,
                timestamp=timestamp,
                tracking_error=tracking_error,
                attitude_error=attitude_error,
                mpc_cost=mpc_cost,
                u_demanded=self.u_past,  # This is the input commanded my the MPC at the current timestep
                velocity = velocity
            )

    def _visualize_prediction(self) -> None:
        """Submit the predicted trajectory; rendering happens after the simulation step."""
        if self.visualization is not None:
            points = [self.gp_mpc.ocp_solver.get(i, "x")[0:3] for i in range(self.ctrl_cfg.N + 1)]
            self.visualization.set_overlay("prediction", points, color=(0, 0, 1, 1), radius=.05)
