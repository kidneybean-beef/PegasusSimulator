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

# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Simple Room"])

        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Basic/clean_cloudy_sky_and_floor.usd") # axis not aligned 
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Basic/display_pedestal.usd") # cannot open
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Basic/display_riser.usd") # cannot open
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Basic/white_void.usd") # axis not aligned 
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Outdoor/Puddles.usd") # axis not aligned
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/LookDev/Decor_Concrete.usd")
        # self.pg.load_environment("/hdd/usd/Assets/Scenes/Templates/Default/DefaultStage.usd")
        self.pg.load_environment("/hdd/usd/AECO_CityDemo/World_CityDemopack.usd") # good but scale not correct
        # self.pg.load_environment("/hdd/usd/AECO_CityMassingDemo/World_CityMassingDemopack.usd")
        # self.pg.load_environment("/hdd/usd/AECO_CityTowerDemo/World_CityTowerDemopack.usd")
        
        # Setup Wind Visualization ---
        self.wind_manager = WindManager(wind_preset=0)
        self._setup_wind_visualization()

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the vehicle 1
        config_multirotor1 = MultirotorConfig()
        config_multirotor1.backends = [NonlinearController(
            results_file=self.curr_dir + "/results/single_statistics_wind.npz",
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
            wind_manager = self.wind_manager
        )]

        self.drone = Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            # [2.3, -1.5, 0.07],
            [2.3,-1.5,90.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )

        # Reset the simulation environment so that all articulations are initialized
        self.world.reset()

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

        # 2. Define the Grid around the drone's spawn point [2.3, -1.5, 0.07]
        # Grid covers X:[0 to 5], Y:[-4 to 1], Z:[0.2 to 3.0]
        x, y, z = np.meshgrid(
            np.linspace(0.0, 5.0, 8), 
            np.linspace(-4.0, 1.0, 8), 
            np.linspace(0.0, 5.0, 5)
        )
        # positions = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # 3. Create simulated wind vectors (e.g., a swirling wind field)
        # You can replace this with your actual wind model or uniform wind: v = [0.5, 0.5, 0]
        # vectors = np.stack([
        #     np.sin(positions[:, 1]),   # Wind X
        #     np.cos(positions[:, 0]),   # Wind Y
        #     np.full(len(positions), 0.1) # Wind Z (slight updraft)
        # ], axis=-1)

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
            colors.append(Gf.Vec3f(float(mag), 0.2, float(1.0 - mag)))
            # colors.append(Gf.Vec4f(float(mag), 0.2, float(1.0 - mag), 0.1))

        # 5. Apply to USD
        self.wind_vis_instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions.tolist()))
        self.wind_vis_instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
        self.wind_vis_instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))
        self.wind_vis_instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices))
        # Create a displayColor Primvar and set it to 'vertex' interpolation (one color per instance)
        color_primvar = UsdGeom.PrimvarsAPI(self.wind_vis_instancer).CreatePrimvar(
            "displayColor", 
            Sdf.ValueTypeNames.Color3fArray, 
            UsdGeom.Tokens.vertex
        )
        # color_primvar.Set(Vt.Vec3fArray(colors))

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
            colors.append(Gf.Vec3f(float(mag), 0.2, float(1.0 - mag)))

        # Update the USD Attributes directly (Notice we don't need to update positions or proto_indices!)
        self.wind_vis_instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
        self.wind_vis_instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))

        # Retrieve the existing color Primvar and update its array
        color_primvar = UsdGeom.PrimvarsAPI(self.wind_vis_instancer).GetPrimvar("displayColor")
        # color_primvar.Set(Vt.Vec3fArray(colors))

    def run(self):
        self.timeline.play()
        camera_offset = np.array([-5.0, 0.0, 1.5])
        target_offset = np.array([0.0, 0.0, 0.5])
        
        while simulation_app.is_running():
            self.world.step(render=True)
            self._update_wind_visualization(self.world.current_time)

            drone_pos = self.drone.state.position
            set_camera_view(eye=drone_pos + camera_offset, target=drone_pos + target_offset)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()