"""Interactive MuJoCo scenes shared by classical environments and RL rollouts."""
import numpy as np

try:
    import viser
    from mjviser import ViserMujocoScene
except ImportError:
    viser = ViserMujocoScene = None


class BrowserViewer:
    def __init__(self, model, data, *, port=8080):
        if viser is None or ViserMujocoScene is None:
            raise ImportError("Browser visualization requires viser and mjviser")
        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        try:
            self.scene = ViserMujocoScene(self.server, model, num_envs=1)
            self.scene.update_from_mjdata(data)
            self.overlays = {}

            @self.server.on_client_connect
            def setup_camera(client):
                client.camera.position = (3, -3, 2)
                client.camera.look_at = tuple(data.xpos[-1])

            actual_port = self.server.get_port()
            print(f"Viewer: http://localhost:{actual_port}\n"
                  f"Remote access: ssh -N -L {actual_port}:127.0.0.1:{actual_port} USER@REMOTE_HOST")
        except BaseException:
            self.server.stop()
            raise

    def update(self, data):
        self.scene.update_from_mjdata(data)

    def set_overlay(self, name, points, color, radius, capsules):
        """Replace one named overlay, retaining other planner/controller layers."""
        import trimesh

        for handle in self.overlays.pop(name, ()):
            handle.remove()
        handles = []
        if len(points):
            handles.append(self.server.scene.add_point_cloud(
                f"/overlays/{name}/points", points=np.asarray(points, dtype=np.float32),
                colors=np.asarray(color[:3]), point_size=2 * radius,
            ))
        for index, (start, end, capsule_radius, rgba) in enumerate(capsules):
            direction = np.asarray(end) - start
            length = np.linalg.norm(direction)
            mesh = trimesh.creation.capsule(radius=capsule_radius, height=float(length))
            if length:
                mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction / length))
            mesh.apply_translation((np.asarray(start) + end) / 2)
            mesh.visual.face_colors = np.asarray(rgba) * 255
            handles.append(self.server.scene.add_mesh_trimesh(f"/overlays/{name}/capsule-{index}", mesh))
        self.overlays[name] = handles

    def close(self):
        self.server.stop()
