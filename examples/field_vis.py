import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Vt

# 1. Setup Stage and Instancer
stage = omni.usd.get_context().get_stage()
instancer_path = "/World/VectorField"
instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

# 2. Create the Prototype (The Arrow/Cone)
# 2. Create the Prototype (An Arrow made of an Xform, Cylinder, and Cone)
prototype_path = "/World/VectorField/Prototypes/Arrow"
arrow_xform = UsdGeom.Xform.Define(stage, prototype_path)

# 2a. Create the Shaft (Cylinder)
shaft_path = prototype_path + "/Shaft"
shaft = UsdGeom.Cylinder.Define(stage, shaft_path)
shaft.GetRadiusAttr().Set(0.05)
shaft.GetHeightAttr().Set(0.1)
shaft.GetAxisAttr().Set("Z")

# FIX: Use XformCommonAPI to safely set the translation
UsdGeom.XformCommonAPI(shaft).SetTranslate(Gf.Vec3d(0, 0, 0))

# 2b. Create the Head (Cone)
head_path = shaft_path + "/Head"
head = UsdGeom.Cone.Define(stage, head_path)
head.GetRadiusAttr().Set(0.15)
head.GetHeightAttr().Set(0.03)
head.GetAxisAttr().Set("Z")

# FIX: Use XformCommonAPI to safely set the translation
UsdGeom.XformCommonAPI(head).SetTranslate(Gf.Vec3d(0, 0, shaft.GetHeightAttr().Get()/2.0+head.GetHeightAttr().Get()/2.0))

# Tell the instancer to use the parent Xform as the prototype
instancer.GetPrototypesRel().AddTarget(prototype_path)

# 3. Generate Grid Data (This is where 'positions' is defined)
grid_size = 5
# Create a 10x10x10 grid from -2 to 2 in X, Y, Z
x, y, z = np.meshgrid(
    np.linspace(-2, 2, grid_size), 
    np.linspace(-2, 2, grid_size), 
    np.linspace(-2, 2, grid_size)
)

# Flatten the grid into a list of (x, y, z) coordinates
positions = np.stack([x, y, z], axis=-1).reshape(-1, 3)

# Create a sample vector field (e.g., a swirling vortex)
# v = [-y, x, 0.1]
vectors = np.stack([-positions[:, 1], positions[:, 0], np.full(len(positions), 0.1)], axis=-1)

# 4. Helper function to calculate rotations (Version-proof for 5.1.0)
def vector_to_quat(v):
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

# 5. Calculate Transforms for the Instancer
scales = []
orientations = []
proto_indices = [0] * len(positions) # 0 maps to our single Cone prototype

for v in vectors:
    mag = np.linalg.norm(v)
    # Scale the Z-axis of the cone based on vector magnitude
    scales.append(Gf.Vec3f(1.0, 1.0, float(mag * 5.0))) 
    orientations.append(vector_to_quat(v))

# 6. Apply Data to USD
instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions.tolist()))
instancer.GetOrientationsAttr().Set(Vt.QuathArray(orientations))
instancer.GetScalesAttr().Set(Vt.Vec3fArray(scales))
instancer.GetProtoIndicesAttr().Set(Vt.IntArray(proto_indices))
