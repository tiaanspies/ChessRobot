import numpy as np
import matplotlib.pyplot as plt
try:
    from chess import FILE_NAMES, RANK_NAMES
except ModuleNotFoundError:
    print("Could not install 'chess' module")
from IK_Solvers.NLinkArm3d import NLinkArm
from IK_Solvers.quintic_polynomials_planner import QuinticPolynomial
import yaml
from path_directories import CONFIG_PATH_KINEMATICS
from path_directories import H_MATRIX_PATH
from Data_analytics import correction_transform
from pathlib import Path

class MotionPlanner():
    def __init__(self, ):
        config = yaml.safe_load(open(CONFIG_PATH_KINEMATICS))['IK_CONFIG']
        self.LIFT = config['lift'] # distance to clear the other pieces in mm
        self.SQUARE_WIDTH = config['chess_square_width'] # width of one board square in mm
        self.BASE_DIST = config['base_dist'] # distance from edge of the board to the robot base in mm
        self.BOARD_HEIGHT = config['board_height'] # height of the board off the ground in mm
        self.GRIP_HEIGHT = config['grip_height'] # how high off the board to grip the pieces in mm
        
        self.l1_params = config['l1_params']
        self.l2_params = config['l2_params']
        self.l3_params = config['l3_params']
        self.l4_params = config['l4_params']

        self.BOARD_WIDTH = 8 * self.SQUARE_WIDTH # total width of the board

        # offset between camera and hinge point that IK solver uses
        c2cpt = config['camera_to_control_pt_offset']
        self.camera_to_control_pt_offset: np.ndarray = np.array([
            [c2cpt['x']],
            [c2cpt['y']],
            [c2cpt['z']]
        ])

        #offset between robot base and aruco board orgin 
        rcs2ccs = config['rcs_to_ccs_offset']
        self.rcs_to_ccs_offset: np.ndarray = np.array([
            [rcs2ccs['x']],
            [rcs2ccs['y']],
            [rcs2ccs['z']]
        ])

        #home position that robot waits at and takes pics of board
        home = config['home_position']
        self.HOME = np.array([
            [home['x']],
            [home['y']],
            [home['z']]
        ])

        self.generate_coords()
        self.initialize_arm()

        try:
            h_list_paths = config['position_compensation_matrices']

            self.h_list = []
            if h_list_paths is not None:
                for path in h_list_paths:
                    full_path = Path(H_MATRIX_PATH, path)
                    print(f"Loading position compensation matrix from {full_path}")
                    self.h_list.append(np.loadtxt(full_path, delimiter=','))

        except FileNotFoundError:
            print("Could not find position compensation matrices. Skipping...")



    def __str__(self):
        return f"Board width: {self.BOARD_WIDTH} mm \nBoard height: {self.BOARD_HEIGHT} mm \nDistance from robot base: {self.BASE_DIST} mm \nSafe lift height: {self.LIFT} mm"

    def generate_coords(self):
        """gives an 3x8x8 ndarray representing the euclidean coordinates of the center of each board square"""
        # zero is at the robot base, which is centered base_dist from the edge of the board
        
        # initialize
        self.board_coords = np.zeros((3,8,8))     
        
        # set z coord of pieces on board
        self.board_coords[2,:,:] = self.BOARD_HEIGHT + self.GRIP_HEIGHT

        # define and set x and y coords of board
        self.file_coords = np.linspace(3.5*self.SQUARE_WIDTH,-3.5*self.SQUARE_WIDTH,8,endpoint=True)
        self.rank_coords = np.linspace(0.5*self.SQUARE_WIDTH,7.5*self.SQUARE_WIDTH,8,endpoint=True) + self.BASE_DIST
        self.board_coords[:2,:,:] = np.array(np.meshgrid(self.file_coords,self.rank_coords))
    
        # define array for coords of captured pieces
        x_row_1 = np.ones(10)*(-self.BOARD_WIDTH/2) - 35
        x_row_2 = np.ones(8)*(-self.BOARD_WIDTH/2) - 65
        y_row_1 = np.linspace(self.BASE_DIST,self.BASE_DIST + self.BOARD_WIDTH, 10,endpoint=True)
        y_row_2 = np.linspace(self.BASE_DIST+60,self.BASE_DIST + self.BOARD_WIDTH, 8,endpoint=True)
        x_coords_storage = np.hstack((x_row_1, x_row_2))
        y_coords_storage = np.hstack((y_row_1, y_row_2))

        self.storage_coords = list(np.vstack((
            x_coords_storage,
            y_coords_storage,
            np.ones(18)*self.GRIP_HEIGHT)
        ).T)
        
        #each coordinate has to be 3,1
        self.storage_coords = [coord.reshape(3,1) for coord in self.storage_coords]

    def initialize_arm(self, param_list=None):
        """initialize inthetasstance of NLinkArm with Denavit-Hartenberg parameters of chess arm"""
        if param_list is None:
            self.param_list = [self.l1_params, self.l2_params, self.l3_params]
        else:
            self.param_list = param_list
        self.chess_arm = NLinkArm(self.param_list)
    
    def get_coords(self, name):
        """gives real-world coordinates in mm based on algebraic move notation (e.g. 'e2e4')"""
        i_file = FILE_NAMES.index(name[0])
        i_rank = RANK_NAMES[::-1].index(name[1])
        return self.board_coords[:,i_rank,i_file].reshape(3,1)

    def generate_path(self, start, goal, cap_sq, step=10):
        """creates a 3xN array of waypoints to and from home, handling captures and lifting over pieces"""
        lift_vector = np.array([0, 0, self.LIFT])
        lift_vector_storage = np.array([0, 0, self.LIFT + self.BOARD_HEIGHT])
        if cap_sq is not None:
            storage = np.array(self.storage_coords.pop(0))
            first_moves = np.hstack((self.line(self.HOME, cap_sq, step),
                                    cap_sq.reshape((3,1)), cap_sq.reshape((3,1)),
                                    self.line(cap_sq, cap_sq + lift_vector, step),
                                    self.line(cap_sq + lift_vector, storage + lift_vector, step),
                                    self.line(storage + lift_vector, storage, step),
                                    storage.reshape((3,1)), storage.reshape((3,1)),
                                    self.line(storage, storage + lift_vector, step),
                                    self.line(storage + lift_vector, start, step)))
        else:
            first_moves = self.line(self.HOME, start, step)
        
        second_moves = np.hstack((start.reshape((3,1)), start.reshape((3,1)),
                                self.line(start, start + lift_vector, step),
                                self.line(start + lift_vector, goal + lift_vector, step),
                                self.line(goal + lift_vector, goal, step),
                                goal.reshape((3,1)), goal.reshape((3,1)),
                                self.line(goal, self.HOME, step)))

        return np.hstack((first_moves, second_moves))

    def generate_quintic_path(self, start, goal, cap_sq=None, rook_start=None, rook_goal=None, step=20):
        """creates a 3xN array of waypoints to and from home, handling captures and lifting over pieces"""

        lift_vector = np.array([[0],[0],[self.LIFT]])
        lift_vector_storage = np.array([[0], [0], [self.LIFT + self.BOARD_HEIGHT]])

        if cap_sq is not None:
            storage = np.array(self.storage_coords.pop(0))
            move1 = self.quintic_line(self.HOME, cap_sq+lift_vector, step) # Move above the piece to be captured
            # Gripper medium open
            move2 = self.quintic_line(cap_sq+lift_vector, cap_sq, step/2) # Move down to piece
            # Close Gripper
            move3 = self.quintic_line(cap_sq, cap_sq + lift_vector, step/2) # Lift piece
            move4 = self.quintic_line(cap_sq + lift_vector, storage + lift_vector_storage, step) # Move to storage
            move5 = self.quintic_line(storage + lift_vector_storage, storage, step) # Place piece in storage
            #Open Gripper
            move6 = self.quintic_line(storage, storage + lift_vector_storage, step) # Lift gripper
            move7 = self.quintic_line(storage + lift_vector_storage, start + lift_vector, step) # Move to capturing piece start
            # Gripper medium open
            move8 = self.quintic_line(start + lift_vector, start, step/2) # Move down to capturing piece

            first_moves = np.hstack((move1, move2, move3, move4, move5, move6, move7, move8))

            # set zeros for all moves where there is no gripper command
            # then a 1 to close gripper and 2 to open
            gripper_medium =  np.size(move1, 1)
            gripper_close = gripper_medium + np.size(move2, 1)
            gripper_open = gripper_close + np.size(move3, 1) + np.size(move4, 1) + np.size(move5, 1)
            gripper_medium2 = gripper_open + np.size(move6, 1) + np.size(move7, 1)

            total_moves = gripper_medium2 + np.size(move8, 1)

            gripper_commands_move1 = np.zeros((1, total_moves))
            gripper_commands_move1[0, gripper_medium] = 2 # medium open
            gripper_commands_move1[0, gripper_close] = 1 # close gripper
            gripper_commands_move1[0, gripper_open] = 3 # open gripper
            gripper_commands_move1[0, gripper_medium2] = 2 # medium open

        else: # No capture
            move1 = self.quintic_line(self.HOME, start+lift_vector, step) # Move above the piece being moved
            #gripper medium open
            move2 = self.quintic_line(start+lift_vector, start, step) # Move down to piece

            first_moves = np.hstack((move1, move2))

            gripper_medium = np.size(move1, 1)

            gripper_commands_move1 = np.zeros((1, np.size(move1, 1) + np.size(move2, 1)))
            gripper_commands_move1[0, gripper_medium] = 2
        
        if rook_start is None:
            #Close Gripper
            move1 = self.quintic_line(start, start+lift_vector, step/2) # Lift piece
            move2 = self.quintic_line(start+lift_vector, goal+lift_vector, step) # Move to goal
            move3 = self.quintic_line(goal+lift_vector, goal, step/2) # Place piece
            #Open Gripper
            move4 = self.quintic_line(goal, goal+lift_vector, step/2) # Lift gripper
            move5 = self.quintic_line(goal+lift_vector, self.HOME, step) # Move to home
            second_moves = np.hstack((move1, move2, move3, move4, move5))

            gripper_close = 0
            gripper_open = np.size(move1, 1) + np.size(move2, 1) + np.size(move3, 1)
            total_moves = gripper_open + np.size(move4, 1)  + np.size(move5, 1)

            gripper_commands_move2 = np.zeros((1, total_moves))
            gripper_commands_move2[0, gripper_close] = 1
            gripper_commands_move2[0, gripper_open] = 3
        else: # normal move and castle
            #Close Gripper
            move1 = self.quintic_line(start, start+lift_vector, step/2) # Lift piece
            move2 = self.quintic_line(start+lift_vector, goal+lift_vector, step) # Move to goal
            move3 = self.quintic_line(goal+lift_vector, goal, step/2) # Place piece
            #Open Gripper
            move4 = self.quintic_line(goal, goal+lift_vector, step/2) # Lift gripper
            move5 = self.quintic_line(goal+lift_vector, rook_start+lift_vector, step) # Move to rook start
            # gripper medium open
            move6 = self.quintic_line(rook_start+lift_vector, rook_start, step/2) # Move down to rook start
            #Close Gripper
            move7 = self.quintic_line(rook_start, rook_start+lift_vector, step/2) # Lift rook
            move8 = self.quintic_line(rook_start+lift_vector, rook_goal+lift_vector, step) # Move rook to goal
            move9 = self.quintic_line(rook_goal+lift_vector, rook_goal, step/2)
            #Open Gripper
            move10 = self.quintic_line(rook_goal, rook_goal+lift_vector, step/2) # Lift gripper
            move11 = self.quintic_line(rook_goal+lift_vector, self.HOME, step)
            second_moves = np.hstack((move1, move2, move3, move4, move5))

            gripper_close1 = 0
            gripper_open1 = np.size(move1, 1) + np.size(move2, 1) + np.size(move3, 1)
            gripper_medium1 = gripper_open1 + np.size(move4, 1) + np.size(move5, 1)
            gripper_close2 = gripper_medium1  + np.size(move6, 1)
            gripper_open2 = gripper_close2 + np.size(move7, 1) + np.size(move8, 1) + np.size(move9, 1)
            total_moves = gripper_open2 + np.size(move10, 1) + np.size(move11, 1)

            gripper_commands_move2 = np.zeros((1, total_moves))
            gripper_commands_move2[0, gripper_close1] = 1 # close gripper
            gripper_commands_move2[0, gripper_open1] = 3 # open gripper
            gripper_commands_move2[0, gripper_medium1] = 2 # medium open
            gripper_commands_move2[0, gripper_close2] = 1 # close gripper
            gripper_commands_move2[0, gripper_open2] = 3 # open gripper
            
        gripper_commands = np.hstack((gripper_commands_move1, gripper_commands_move2))

        return np.hstack((first_moves, second_moves)), gripper_commands

    def draw_flat_cube(self, z, x_neg, x_pos, y_neg, y_pos):
        """Draws a flat cube"""
        step = 10
        path = np.hstack([
            self.quintic_line(self.HOME, np.array([x_neg, y_neg, z]), step),
            self.quintic_line(np.array([x_neg, y_neg, z]), np.array([x_pos, y_neg, z]), step),
            self.quintic_line(np.array([x_pos, y_neg, z]), np.array([x_pos, y_pos, z]), step),
            self.quintic_line(np.array([x_pos, y_pos, z]), np.array([x_neg, y_pos, z]), step),
            self.quintic_line(np.array([x_neg, y_pos, z]), np.array([x_neg, y_neg, z]), step),
            self.quintic_line(np.array([x_neg, y_neg, z]), self.HOME, step)
        ])

        return path

    def draw_cube(self, v, slice_num):
        #NUM LAYERS MUST BE EVEN
        NUM_LAYERS = 8

        #NUM ROWS MUST BE EVEN
        NUM_ROWS = 10
        path = np.zeros((3, 0))
        layers = np.linspace(v["top"], v["bottom"], NUM_LAYERS, endpoint=True)

        for j in range(0, NUM_LAYERS, 2):
            layer_zig_zag = np.zeros((3, 0))
            rows = np.linspace(v["close"], v["far"], NUM_ROWS, endpoint=True)

            for i in range(0, NUM_ROWS, 2):
                start_dir_right = np.array([v["left"], rows[i], layers[j]]).reshape(3, 1)
                end_dir_right = np.array([v["right"], rows[i], layers[j]]).reshape(3, 1)

                start_dir_left = np.array([v["right"], rows[i+1], layers[j]]).reshape(3, 1)
                end_dir_left = np.array([v["left"], rows[i+1], layers[j]]).reshape(3, 1)

                layer_zig_zag = np.hstack([layer_zig_zag, 
                    self.line(start_dir_right, end_dir_right, 20),
                    self.line(end_dir_right, start_dir_left, 20),
                    self.line(start_dir_left, end_dir_left, 20),
                ])

            rows = np.linspace(v["far"], v["close"], NUM_ROWS, endpoint=True)
            for i in range(0, NUM_ROWS, 2):
                start_dir_right = np.array([v["left"], rows[i], layers[j+1]]).reshape(3, 1)
                end_dir_right = np.array([v["right"], rows[i], layers[j+1]]).reshape(3, 1)

                start_dir_left = np.array([v["right"], rows[i+1], layers[j+1]]).reshape(3, 1)
                end_dir_left = np.array([v["left"], rows[i+1], layers[j+1]]).reshape(3, 1)

                layer_zig_zag = np.hstack([layer_zig_zag, 
                    self.line(start_dir_right, end_dir_right, 20),
                    self.line(end_dir_right, start_dir_left, 20),
                    self.line(start_dir_left, end_dir_left, 20),
                ])         
            
            path = np.hstack([path, layer_zig_zag])

        return path

    def inverse_kinematics(self, path, apply_compensation):
        """generates a 4xN list of joint angles from a 3xN list of waypoints"""

        if apply_compensation:
            path = correction_transform.project_points_quad_multiple(path, self.h_list)
 
        # execute IK on each point in the path
        num_waypoints = np.size(path,1)
        theta_path = np.zeros((3,num_waypoints))

        # get the first point
        waypoint = np.hstack((path[:,0],0,0,0))
        theta_path[:, 0] = self.chess_arm.inverse_kinematics(waypoint)

        # calculate the rest with gradient descent
        for waypoint_idx in range(1, num_waypoints):
            waypoint = np.hstack((path[:,waypoint_idx]))
            theta_path[:,waypoint_idx] = self.chess_arm.inverse_kinematics_grad_descent(waypoint)
            if waypoint_idx % 10 == 0:
                print(f"Waypoint {waypoint_idx}/{num_waypoints} complete.")
        return theta_path

    def forward_kinematics(self, thetas):
        num_waypoints = np.size(thetas,1)
        xpath = np.zeros((3,num_waypoints))
        for point in range(num_waypoints):
            theta = list(thetas[:,point])
            self.chess_arm.set_joint_angles(theta)
            xpath[:,point] = self.chess_arm.forward_kinematics()[:3]
        return xpath

    def plot_board(self, ax):
        """plots the given path along with a representation of the chess board"""
        file_lines = np.linspace(-4*self.SQUARE_WIDTH,4*self.SQUARE_WIDTH,9,endpoint=True)
        rank_lines = np.linspace(8*self.SQUARE_WIDTH,0,9,endpoint=True) + self.BASE_DIST
        X,Y = np.meshgrid(file_lines,rank_lines)
        Z = np.ones_like(X) * self.BOARD_HEIGHT
        ax.plot_wireframe(X,Y,Z, color="k")
        
    def plot_robot(self, thetas, path=None):
        n_steps = np.size(thetas,1)
        verts = np.zeros((n_steps, 3, len(self.chess_arm.link_list) + 1))
        for i_step in range(n_steps):
            self.chess_arm.set_joint_angles(thetas[:,i_step])
            verts[i_step,:,:] = self.chess_arm.get_vertices()
        
        ax = plt.axes(projection = '3d')
        
        file_lines = np.linspace(-4*self.SQUARE_WIDTH,4*self.SQUARE_WIDTH,9,endpoint=True)
        rank_lines = np.linspace(8*self.SQUARE_WIDTH,0,9,endpoint=True) + self.BASE_DIST
        X,Y = np.meshgrid(file_lines,rank_lines)
        Z = np.ones_like(X) * self.BOARD_HEIGHT
        board = ax.plot_wireframe(X,Y,Z, color="r")

        base = ax.plot([0], [0], [0], "o", ms=10, mfc='k', mec='k')

        arm = ax.plot(verts[0,0,:], verts[0,1,:], verts[0,2,:], "o-", 
                            color="#00aa00", ms=10, mfc='k', mec='k')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(-200, 200)
        ax.set_ylim(0,500)
        ax.set_zlim(0, 300)

        for step in range(n_steps):
            for link in arm:
                link.remove()
            arm = ax.plot(verts[step,0,:], verts[step,1,:], verts[step,2,:], "o-", 
                            color="#00aa00", lw=10, ms=10, mfc='k', mec='k')
            if path is not None:
                waypoint = ax.plot(path[0,:step],path[1,:step],path[2,:step], 'bo', ms=1)
            plt.draw()
            plt.pause(.01)
        plt.show()

    def quintic_line(self, start, goal, avg_step):
        """Creates a 3xN nparray of 3D waypoints between start and goal using a quintec polynomial"""
        
        dist = np.linalg.norm(goal - start)
        n_steps = int(dist // avg_step)
        xqnt = QuinticPolynomial(start[0, 0],goal[0, 0],n_steps)
        yqnt = QuinticPolynomial(start[1, 0],goal[1, 0],n_steps)
        zqnt = QuinticPolynomial(start[2, 0],goal[2, 0],n_steps)

        path = np.zeros((3,n_steps))
        for step in range(n_steps):
            path[:,step] = [xqnt.calc_point(step), yqnt.calc_point(step), zqnt.calc_point(step)]
        
        return path
    
    @staticmethod
    def line(start, goal, step):
        """Creates a 3xN nparray of 3D waypoints roughly 'step' distance apart between two 3D points"""
        dist = np.linalg.norm(goal - start)
        n_steps = int(dist // step)
        x_points = np.linspace(start[0], goal[0], n_steps, endpoint=True).T
        y_points = np.linspace(start[1], goal[1], n_steps, endpoint=True).T
        z_points = np.linspace(start[2], goal[2], n_steps, endpoint=True).T
        
        line = np.vstack((x_points, y_points, z_points))
        return line

    def control_pt_to_camera_pos(self, control_pt_coords: np.ndarray):
        """Gets the position of the camera when given the control pt position"""
        #TODO: Use more complex method to project backwards from camera.
        # this method assumes the camera is perfectly vertical.

        control_pt_coords

        camera_coords = control_pt_coords - self.camera_to_control_pt_offset

        return camera_coords

    def camera_to_control_pt_pos(self, camera_coords:np.ndarray):
        """Gets the control point position when given the camera position."""

        if (camera_coords == np.nan).any():
            return camera_coords
        
        control_pt_coords = camera_coords + self.camera_to_control_pt_offset

        return control_pt_coords

    def rcs_to_ccs(self, rcs_coords:np.ndarray):
        """Converts from robot coordinate system to camera coordinate system."""
        
        if (rcs_coords == np.nan).any():
            return rcs_coords

        ccs_coords = rcs_coords + self.rcs_to_ccs_offset

        return ccs_coords

    def ccs_to_rcs(self, ccs_coords:np.ndarray):
        """Converts from camera coordinate system to robot coordinate system."""

        if (ccs_coords == np.nan).any():
            return ccs_coords
        
        rcs_coords = ccs_coords - self.rcs_to_ccs_offset

        return rcs_coords
    