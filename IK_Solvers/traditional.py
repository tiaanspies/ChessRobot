import numpy as np
import matplotlib.pyplot as plt
from chess import FILE_NAMES, RANK_NAMES
from IK_Solvers.NLinkArm3d import NLinkArm

class ChessMoves():
    
    def __init__(self, lift=50, square_width=30, base_dist=100, board_height=10, piece_height=50, L1=250, L2=250):
        self.LIFT = lift # distance to clear the other pieces in mm
        self.SQUARE_WIDTH = square_width # width of one board square in mm
        self.BASE_DIST = base_dist # distance from edge of the board to the robot base in mm
        self.BOARD_HEIGHT = board_height # height of the board off the ground in mm
        self.PIECE_HEIGHT = piece_height # height of the pieces off the board in mm
        self.L1 = L1 # length of the first link in mm
        self.L2 = L2 # length of the second link in mm
        self.BOARD_WIDTH = 8 * self.SQUARE_WIDTH # total width of the board
        self.HOME = np.array([0, self.BASE_DIST, 1.5*self.BOARD_WIDTH]) # location of home for the robot arm between moves

        self.generate_coords()
        self.initialize_arm()

    def __str__(self):
        return f"Board width: {self.BOARD_WIDTH} mm \nBoard height: {self.BOARD_HEIGHT} mm \nDistance from robot base: {self.BASE_DIST} mm \nSafe lift height: {self.LIFT} mm"

    def generate_coords(self):
        """gives an 3x8x8 ndarray representing the euclidean coordinates of the center of each board square"""
        # zero is at the robot base, which is centered base_dist from the edge of the board
        
        # initialize
        self.board_coords = np.zeros((3,8,8))     
        
        # set z coord of pieces on board
        self.board_coords[2,:,:] = self.BOARD_HEIGHT + self.PIECE_HEIGHT

        # define and set x and y coords of board
        self.file_coords = np.linspace(-3.5*self.SQUARE_WIDTH,3.5*self.SQUARE_WIDTH,8,endpoint=True)
        self.rank_coords = np.linspace(7.5*self.SQUARE_WIDTH,0.5*self.SQUARE_WIDTH,8,endpoint=True) + self.BASE_DIST
        self.board_coords[:2,:,:] = np.array(np.meshgrid(self.file_coords,self.rank_coords))
    
        # define array for coords of captured pieces
        self.storage_coords = list(np.vstack((np.linspace(-self.BOARD_WIDTH/2,self.BOARD_WIDTH/2,15,endpoint=True),
                                np.ones(15)*(self.BASE_DIST - self.SQUARE_WIDTH),
                                np.zeros(15))).T)

    def initialize_arm(self):
        """initialize instance of NLinkArm with Denavit-Hartenberg parameters of chess arm"""
        l1_params = [0, np.pi/2, 0, 0]
        l2_params = [0, 0, self.L1, 0]
        l3_params = [0, 0, self.L2, 0]
        l4_params = [np.pi,0,0,0]
        self.param_list = [l1_params, l2_params, l3_params, l4_params]
        self.chess_arm = NLinkArm(self.param_list)

    def get_coords(self, name):
        """gives real-world coordinates in mm based on algebraic move notation (e.g. 'e2e4')"""
        i_file = FILE_NAMES.index(name[0])
        i_rank = RANK_NAMES[::-1].index(name[1])
        return self.board_coords[:,i_rank,i_file]

    def generate_path(self, start, goal, cap_sq, step=10):
        """creates a 3xN array of waypoints to and from home, handling captures and lifting over pieces"""
        lift_vector = np.array([0,0,self.LIFT])
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

    def inverse_kinematics(self, path):
        """generates a 4xN list of joint angles from a 3xN list of waypoints"""
        # execute IK on each point in the path
        num_waypoints = np.size(path,1)
        theta_path = np.zeros((4,num_waypoints))
        for waypoint_idx in range(num_waypoints):
            waypoint = list(np.hstack((path[:,waypoint_idx],0,0,0)))
            theta_path[:,waypoint_idx] = self.chess_arm.inverse_kinematics(waypoint)
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

    @staticmethod
    def line(start, goal, step):
        """Creates a 3xN nparray of 3D waypoints roughly 'step' distance apart between two 3D points"""
        dist = np.linalg.norm(goal - start)
        n_steps = int(dist // step)
        x_points = np.linspace(start[0], goal[0], n_steps, endpoint=False)
        y_points = np.linspace(start[1], goal[1], n_steps, endpoint=False)
        z_points = np.linspace(start[2], goal[2], n_steps, endpoint=False)
        
        return np.vstack((x_points, y_points, z_points))