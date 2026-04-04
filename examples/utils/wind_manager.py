import numpy as np
from scipy.interpolate import RegularGridInterpolator

class WindManager:
    def __init__(self, wind_preset=0, drone_hover_pos=[0.0, 0.0, 5.0]):
        # 1. Define the grid axes
        # self.x_coords = np.linspace(-5.0, 5.0, 10)
        # self.y_coords = np.linspace(-5.0, 5.0, 10)
        # self.z_coords = np.linspace(90.0, 100.0, 10)
        self.drone_hover_pos = drone_hover_pos
        self.wind_r = 10.0
        self.x_coords = np.linspace(self.drone_hover_pos[0]-self.wind_r, self.drone_hover_pos[0]+self.wind_r, 10)
        self.y_coords = np.linspace(self.drone_hover_pos[1]-self.wind_r, self.drone_hover_pos[1]+self.wind_r, 10)
        self.z_coords = np.linspace(self.drone_hover_pos[2]-self.wind_r, self.drone_hover_pos[2]+self.wind_r, 10)

        # 2. Create the 3D grid (using 'ij' indexing is required for RegularGridInterpolator)
        X, Y, Z = np.meshgrid(self.x_coords, self.y_coords, self.z_coords, indexing='ij')
        self.X = X
        self.Y = Y
        self.Z = Z
        self.positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

        # 3. Define the wind vectors mathematically
        # self.U = 2.0*np.sin(Y)                     # Wind X
        # self.V = 2It acts like a person turning their head. It will always spin the camera on its own local axis, and you cannot natively change the Right Mouse Button to orbit a parent object just using the UI..0*np.cos(X)                     # Wind Y
        # self.W = np.full_like(X, 0.5)          # Wind Z (slight updraft)

        # # uniform wind
        # if wind_preset == "uniform":
        #     self.U = np.ones_like(X)                    # Wind X
        #     self.V = np.ones_like(X)                   # Wind Y
        #     self.W = np.full_like(X, 0.5)          # Wind Z (slight updraft)
        if wind_preset == 0:
            def wind_generator(X, Y, Z, t):
                U = np.ones_like(X)       
                # V = np.cos(X - (t * 5.0)) 
                V = np.ones_like(X) * np.cos(t * 10.0)*0.5
                W = np.full_like(X, 0.1 + 0.5 * np.sin(t * 10.0)) # Pulsing updraft
                return U, V, W
            self.wind_generator = wind_generator 
        elif wind_preset == 1:
            def wind_generator(X, Y, Z, t):
                U = np.zeros_like(X)                    # Wind X
                V = np.ones_like(X)                   # Wind Y
                W = np.full_like(X, 0.1 + 0.5 * np.sin(5*t * 2.0)) # Pulsing updraft          # Wind Z (slight updraft)
                return U, V, W
            self.wind_generator = wind_generator
        elif wind_preset == 2:
            def wind_generator(X, Y, Z, t):
                U = np.sin(Y*0.2)                     # Wind X
                V = np.cos(X*0.2)                     # Wind Y
                # W = np.full_like(X, 0.2)          # Wind Z (slight updraft)
                W = np.sin(Z*0.1)
                return U, V, W
            self.wind_generator = wind_generator                              

        # 4. Create the 3D interpolators for X, Y, and Z wind components
        # bounds_error=False and fill_value=0.0 means if the drone flies outside the grid, wind is 0
        # self.interp_u = RegularGridInterpolator((self.x_coords, self.y_coords, self.z_coords), self.U, bounds_error=False, fill_value=0.0)
        # self.interp_v = RegularGridInterpolator((self.x_coords, self.y_coords, self.z_coords), self.V, bounds_error=False, fill_value=0.0)
        # self.interp_w = RegularGridInterpolator((self.x_coords, self.y_coords, self.z_coords), self.W, bounds_error=False, fill_value=0.0)

    def get_wind_at_position(self, position, t):
        """
        Takes a [x, y, z] position and returns the interpolated wind vector [u, v, w]
        """
        # RegularGridInterpolator expects a shape of points, so we wrap it in a list
        # u = self.interp_u([position])[0]
        # v = self.interp_v([position])[0]
        # w = self.interp_w([position])[0]
        # Example time-varying math: The wind swirls and shifts over time
        x, y, z = position
        # u = np.sin(y + t)       
        # v = np.cos(x - (t * 0.5)) 
        # w = 0.1 + 0.05 * np.sin(t * 2.0) # Pulsing updraft        
        u, v, w = self.wind_generator(x, y, z, t)

        return np.array([u, v, w])

    def get_visualization_data(self, t):
        """
        Helper method to export the flattened positions and vectors for the USD PointInstancer
        """
        # Flatten the arrays for USD Instancer
        # positions = np.stack([self.X.flatten(), self.Y.flatten(), self.Z.flatten()], axis=-1)
        # U = np.sin(self.Y + t)
        # V = np.cos(self.X - (t * 0.5))
        # W = np.full_like(self.X, 0.1 + 0.05 * np.sin(t * 2.0))
        U, V, W = self.wind_generator(self.X, self.Y, self.Z, t)
        vectors = np.stack([U.flatten(), V.flatten(), W.flatten()], axis=-1)
        return self.positions, vectors