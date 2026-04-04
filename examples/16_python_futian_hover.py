#!/usr/bin/env python
"""
| File: 4_python_single_vehicle.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"headless": False})


from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.graph.window.action")
enable_extension("omni.graph.ui")

from omni.isaac.core.utils.prims import add_reference_to_stage
import omni.kit.viewport.utility as vp_util

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
import omni.usd
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Import the custom python control backend
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/utils')
from nonlinear_controller import NonlinearController
from wind_manager import WindManager

# Auxiliary math and USD modules
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Gf, Vt, Sdf, UsdShade
from omni.isaac.core.utils.viewports import set_camera_view
import omni.kit.viewport.utility as vp_util
from pxr import UsdGeom, Usd
from omni.isaac.core.prims.xform_prim import XFormPrim

# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Futian environment
        add_reference_to_stage(usd_path="/hdd/usd/Futian/upperhills.usd", prim_path="/World/upperhills")
        add_reference_to_stage(usd_path="/hdd/usd/Futian/lianhuashan_landingplatform.usd", prim_path="/World/lianhuashan_landingplatform")
        add_reference_to_stage(usd_path="/hdd/usd/Futian/futian_terrain_lianhuashan.usd", prim_path="/World/futian_terrain_lianhuashan")
        add_reference_to_stage(usd_path="/hdd/usd/Futian/futian_highspeedstation.usd", prim_path="/World/futian_highspeedstation")
        add_reference_to_stage(usd_path="/hdd/usd/Futian/futian_buildings.usd", prim_path="/World/futian_buildings")
        add_reference_to_stage(usd_path="/hdd/usd/Futian/bijiashan_park.usd", prim_path="/World/bijiashan_park")

        # 
        #       
        # Setup Wind Visualization ---
        self.wind_manager = WindManager(wind_preset=2, drone_hover_pos=
                                        [1770.5291748046875, -3440.539306640625, 150.35646057128906] # scene 1
                                        )
        self._setup_wind_visualization()

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the vehicle 1
        config_multirotor1 = MultirotorConfig()
        config_multirotor1.backends = [NonlinearController(
            # trajectory_file=self.curr_dir + "/trajectories/upperhills_rev.csv", # path following

            results_file=self.curr_dir + "/results/single_statistics.npz",
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
            wind_manager = self.wind_manager
        )]

        x, y, z = self.wind_manager.drone_hover_pos
        self.drone = Multirotor(
            stage_prefix="/World/quadrotor1",
            usd_file=ROBOTS['Iris'],
            vehicle_id=0,
            init_pos=[x - self.wind_manager.wind_r/2.0, y - self.wind_manager.wind_r, z - self.wind_manager.wind_r/2.0], # hover-in-the-wind test, close to buildings
            init_orientation=Rotation.from_euler("XYZ", [0.0, 0.0, 90.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )

        # Reset the simulation environment so that all articulations are initialized
        self.world.reset()

        add_reference_to_stage(
                    usd_path=self.curr_dir + "/CameraRig7.usd", 
                    prim_path="/World"
                )
        # 2. Dynamically link the Action Graph to the Drone Body
        stage = self.world.stage
        
        # NOTE: Make sure this path matches the exact name of the node in your Stage tree!
        pose_node_path = "/World/CameraRig/ActionGraph/isaac_read_world_pose" 
        pose_node_prim = stage.GetPrimAtPath(pose_node_path)
        
        if pose_node_prim.IsValid():
            # Inject the target relationship into the node
            pose_node_prim.GetRelationship("inputs:prim").SetTargets([Sdf.Path("/World/quadrotor1/body")])
        else:
            carb.log_error(f"Could not find the Pose node at {pose_node_path}!")        

    def _vector_to_quat(self, v):
        """Helper to convert a wind vector to a USD Quaternion"""
        mag = np.linalg.norm(v)
        if mag < 1e-6:
            return Gf.Quath(1.0, 0.0, 0.0, 0.0) 
            
        v_norm = v / mag
        z_axis = np.array([0.0, 0.0, 1.0])
        dot = np.dot(z_axis, v_norm)
        
        if dot > 0.9999:
            return Gf.Quath(1.0, 0.0, 0.0, 0.0) 
        elif dot < -0.9999:
            return Gf.Quath(0.0, 1.0, 0.0, 0.0) 
            
        cross = np.cross(z_axis, v_norm)
        q = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
        q = q / np.linalg.norm(q) 
        
        return Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _setup_wind_visualization(self):
        """Generates the USD PointInstancer and arrow prototypes for the wind field"""
        stage = omni.usd.get_context().get_stage()
        instancer_path = "/World/WindField"
        self.wind_vis_instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

        # 1. Create Arrow Prototype
        prototype_path = "/World/WindField/Prototypes/Arrow"
        arrow_xform = UsdGeom.Xform.Define(stage, prototype_path)

        material_path = prototype_path + "/TransparentMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")

        # 1. Setup the main surface shader
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.3) 
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0) 
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        
        # 2. Setup the Primvar Reader with the CORRECT ID
        color_reader = UsdShade.Shader.Define(stage, material_path + "/ColorReader")
        color_reader.CreateIdAttr("UsdPrimvarReader_float3") # <-- This is the standard USD fix!
        color_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        # Provide a fallback color just in case it takes a frame to load
        color_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        color_reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        
        # 3. Connect the reader's output to the shader's diffuseColor input
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(color_reader.ConnectableAPI(), "result")
        
        # 4. Connect and Bind
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(arrow_xform).Bind(material)

        # Shaft
        shaft_path = prototype_path + "/Shaft"
        shaft = UsdGeom.Cylinder.Define(stage, shaft_path)
        shaft.GetRadiusAttr().Set(0.01)
        shaft.GetHeightAttr().Set(0.1)
        shaft.GetAxisAttr().Set("Z")
        UsdGeom.XformCommonAPI(shaft).SetTranslate(Gf.Vec3d(0, 0, 0.05))

        # Head
        head_path = prototype_path + "/Head"
        head = UsdGeom.Cone.Define(stage, head_path)
        head.GetRadiusAttr().Set(0.03)
        head.GetHeightAttr().Set(0.05)
        head.GetAxisAttr().Set("Z")
        UsdGeom.XformCommonAPI(head).SetTranslate(Gf.Vec3d(0, 0, 0.125))

        self.wind_vis_instancer.GetPrototypesRel().AddTarget(prototype_path)

        positions, vectors = self.wind_manager.get_visualization_data(t=0.0)

        # 4. Process vectors into Transforms
        scales = []
        orientations = []
        colors = []
        proto_indices = [0] * len(positions)

        for v in vectors:
            mag = np.linalg.norm(v)
            scales.append(Gf.Vec3f(1.0, 1.0, float(mag*2.0))) # Scale arrow length by wind speed
            orientations.append(self._vector_to_quat(v))
            # Optional: Color gradient (blue to red based on magnitude)
            # colors.append(Gf.Vec3f(float(mag), 0.2, float(1.0 - mag)))

        # 5. Apply to USD
        self.wind_vis_instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions.tolist()))
        self.wind_vis_instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
        self.wind_vis_instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))
        self.wind_vis_instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices))

    def _update_wind_visualization(self, t):
        """Updates the wind arrows' direction, scale, and color for a given time t"""
        
        # Get the new time-varying vectors from the manager
        _, vectors = self.wind_manager.get_visualization_data(t)

        scales = []
        orientations = []
        colors = []

        for v in vectors:
            mag = np.linalg.norm(v)
            scales.append(Gf.Vec3f(1.0, 1.0, float(mag * 2.0)))
            orientations.append(self._vector_to_quat(v))
            # colors.append(Gf.Vec3f(float(mag), 0.2, float(1.0 - mag)))

        # Update the USD Attributes directly (Notice we don't need to update positions or proto_indices!)
        self.wind_vis_instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
        self.wind_vis_instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))

    def run(self):
        self.timeline.play()
        
        while simulation_app.is_running():
            self.world.step(render=True)
            self._update_wind_visualization(self.world.current_time)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()