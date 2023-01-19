import numpy as np
import matplotlib.pyplot as plt
import chess
from itertools import permutations
from NLinkArm3d import NLinkArm
from keras import Input, Model
from keras.layers import Dense, Rescaling
import time

class ChessMoves():
    
    def __init__(self, lift=50, square_width=30, base_dist=100, board_height=10, piece_height=50):
        self.LIFT = lift # distance to clear the other pieces in mm
        self.SQUARE_WIDTH = square_width # width of one board square in mm
        self.BASE_DIST = base_dist # distance from edge of the board to the robot base in mm
        self.BOARD_HEIGHT = board_height # height of the board off the ground in mm
        self.PIECE_HEIGHT = piece_height # height of the pieces off the board in mm

        self.BOARD_WIDTH = 8 * self.SQUARE_WIDTH # total width of the board
        self.HOME = np.array([0, base_dist, 1.5*self.BOARD_WIDTH]) # location of home for the robot arm between moves

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
    
    def move_to_coords(self, move):
        """gives real-world coordinates in mm based on algebraic move notation (e.g. 'e2e4')"""
        i_startfile = chess.FILE_NAMES.index(move[0])
        i_startrank = chess.RANK_NAMES[::-1].index(move[1])
        i_goalfile = chess.FILE_NAMES.index(move[2])
        i_goalrank = chess.RANK_NAMES[::-1].index(move[3])
        return self.board_coords[:,i_startrank,i_startfile], self.board_coords[:,i_goalrank,i_goalfile]

    def generate_full_path(self, start, goal, tot_steps):
        """creates a 3xN array of waypoints from home and back to the 3x1 start and 3x1 goal, lifting over obstacles"""
        home_steps = tot_steps//4
        lift_steps = tot_steps//8
        go_steps = tot_steps - home_steps * 2 - lift_steps * 2 - 1
        lift_vector = np.array([0,0,self.LIFT])
        return np.hstack((self.extrapolate_points(self.HOME, start, home_steps),
                            self.extrapolate_points(start, start + lift_vector, lift_steps),
                            self.extrapolate_points(start + lift_vector, goal + lift_vector, go_steps),
                            self.extrapolate_points(goal + lift_vector, goal, lift_steps),
                            self.extrapolate_points(goal, self.HOME, home_steps),
                            np.reshape(self.HOME,(3,1))))
        
    def generate_lift_path(self, start, goal, tot_steps):
        """creates a 3xN array of waypoints from the 3x1 start to the 3x1 goal, lifting over obstacles"""
        lift_steps = tot_steps//4
        go_steps = tot_steps - lift_steps * 2 - 1
        lift_vector = np.array([0,0,self.LIFT])
        return np.hstack((self.extrapolate_points(start, start + lift_vector, lift_steps),
                            self.extrapolate_points(start + lift_vector, goal + lift_vector, go_steps),
                            self.extrapolate_points(goal + lift_vector, goal, lift_steps),
                            np.reshape(goal,(3,1))))

    def generate_flat_path(self, start, goal, tot_steps):
        """creates a 3xN array of waypoints from the 3x1 start to the 3x1 goal, lifting over obstacles"""
        go_steps = tot_steps - 1      
        return np.hstack((self.extrapolate_points(start, goal, go_steps), np.reshape(goal,(3,1))))

    def initialize_arm(self, L1=250, L2=250):
        """initialize instance of NLinkArm with Denavit-Hartenberg parameters of chess arm"""
        l1_params = [0, np.pi/2, 0, 0]
        l2_params = [0, 0, L1, 0]
        l3_params = [0, 0, L2, 0]
        l4_params = [np.pi,0,0,0]
        self.param_list = [l1_params, l2_params, l3_params, l4_params]
        self.chess_arm = NLinkArm(self.param_list)

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
    def extrapolate_points(start, goal, n_steps):
        """Creates a 3xN nparray of 3D waypoints roughly 'step' distance apart between two 3D points"""
        x_points = np.linspace(start[0], goal[0], n_steps, endpoint=False)
        y_points = np.linspace(start[1], goal[1], n_steps, endpoint=False)
        z_points = np.linspace(start[2], goal[2], n_steps, endpoint=False)
        
        return np.vstack((x_points, y_points, z_points))

def generate_training_data(path_type, n_steps):
    '''generate training data based on what type of path and the number of steps wanted'''
    if path_type.lower() not in ["full", "lift", "flat"]:
        raise ValueError

    cm = ChessMoves()
    
    # list of dictionaries. Each dictionary represents a possible move and will contain the coordinates of the start and goal, path waypoints, and q joint angles
    move_idxs = permutations(range(64),2)
    all_moves = [{"name": chess.square_name(move[0]) + chess.square_name(move[1]), "start":None, "goal":None, "path":None, "q":None} for move in move_idxs]
    all_moves = np.array(all_moves)
    
    for move in all_moves:
        # generate the coordinates of the start and goal squares
        move['start'], move['goal'] = cm.move_to_coords(move['name'])

        # generate a path between them, based on the path type wanted
        if path_type == 'full':
            move['path'] = cm.generate_full_path(move['start'], move['goal'], n_steps)
        elif path_type == 'lift':
            move['path'] = cm.generate_lift_path(move['start'], move['goal'], n_steps)
        elif path_type == 'flat':
            move['path'] = cm.generate_flat_path(move['start'], move['goal'], n_steps)

        # generate the joint angles for a given path
        move['q'] = cm.inverse_kinematics(move['path'])
        print(f"{move['name']} finished")
    
    # save the data
    np.save('all_lift.npy',all_moves)

### Functions for Single Network architecture ###
def build_model(n_steps):
    '''builds a model that can work for the path or the inverse kinematics'''
    dim = 4 * n_steps
    
    inputs = Input(shape=(6,))
    x = Rescaling(scale=1/500)(inputs)
    x = Dense(dim*200, activation='tanh')(x)
    #x = Dense(dim*200, activation='tanh')(x)
    #x = Dense(dim*100, activation='sigmoid')(x)
    #x = Dense(dim*50, activation='tanh')(x)
    outputs = Dense(dim, activation='linear')(x)
    #outputs = Rescaling(scale=500)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="rmsprop", loss='mse', metrics=['accuracy'])
    model.summary()
       
    return model

def preprocess_data(filename):
    """reads in the numpy arrays of dictionaries and produces training and validation data for going straight from start and goal to path or angles"""
    # get the data and shuffle it
    all_moves = np.load(filename, allow_pickle=True)
    n_moves = len(all_moves)
    n_steps = np.size(all_moves[0]['q'],1)
    n_dim = np.size(all_moves[0]['q'],0)
    all_moves = np.take(all_moves,np.random.permutation(n_moves),axis=0,out=all_moves)

    # convert it into inputs and outputs for the neural network
    input = np.zeros((n_moves,6))
    output = np.zeros((n_moves,n_steps*n_dim))
    for i_move in range(n_moves):
        input[i_move,:3] = all_moves[i_move]["start"]
        input[i_move,3:] = all_moves[i_move]["goal"]
        output[i_move,:] = all_moves[i_move]['q'].flatten(order='F')

    # divide it into training, validation, and test sets
    val_num = n_moves//4
    x = input[val_num:-val_num,:]
    y = output[val_num:-val_num,:]
    x_val = input[-val_num:,:]
    y_val = output[-val_num:,:]
    x_test = input[:val_num,:]
    y_test = output[:val_num,:]
    names_test = np.empty(val_num, dtype=object)
    for i_name in range(val_num):
        names_test[i_name] = all_moves[i_name]['name']
    return n_steps, x, y, x_val, y_val, x_test, y_test, names_test

def test_together(chess_net, x, y, n_steps, names, test_num):
        
    cm2 = ChessMoves()
    dim = 4
    
    for test_idx in range(test_num):
        test_in = x[test_idx,:].reshape((1,6))
        ref_thetas = y[test_idx,:].reshape((dim,n_steps), order='F')
        
        # neural net path
        nn_thetas = chess_net.predict(test_in).reshape((dim,n_steps), order='F')
        ref_path = cm2.forward_kinematics(ref_thetas)
        nn_path = cm2.forward_kinematics(nn_thetas)
        
        fig = plt.figure()
        ax = plt.subplot(projection='3d')
        name = names[test_idx]
        plt.title(f'Single Network Path Results: {name}')
        cm2.plot_board(ax)
        ax.scatter3D(ref_path[0],ref_path[1],ref_path[2], label='Reference Path')
        ax.scatter3D(nn_path[0],nn_path[1],nn_path[2], label='NN path')
        plt.legend()
        plt.show()

    cm2.plot_robot(nn_thetas, ref_path)
 
def time_test_together(model, x, names, n_steps):
    cm3 = ChessMoves()
    trad_times = np.zeros((3,1))
    nn_times = np.zeros((3,1))
    for j in range(3):
        start_trad = time.time()
        print('Traditionally solving for...')
        for i in range(10):
            move_name = names[i]
            print(move_name)
            start, goal = cm3.move_to_coords(move_name)
            path = cm3.generate_flat_path(start,goal,n_steps)
            thetas = cm3.inverse_kinematics(path)
        trad_time = time.time() - start_trad
        print(f"Traditional time: {trad_time}")
        trad_times[j] = trad_time

        print('Neural Net solving for...')
        start_nn = time.time()
        for i in range(10):
            move_name = names[i]
            print(move_name)
            start, goal = cm3.move_to_coords(move_name)
            input = np.zeros((1,6))
            input[0,:3] = start
            input[0,3:] = goal
            thetas = model.predict(input).reshape((4,n_steps), order='F')
        nn_time = time.time() - start_nn
        print(f"Neural Net time: {nn_time}")
        nn_times[j] = nn_time
    
    print(f'Average Traditional time: {np.mean(trad_times)}')
    print(f'Average Neural Net time: {np.mean(nn_times)}')

def main_together():
    '''
    # GENERATE DATA
    path_type = 'lift'
    n_steps = 20
    print(f"generating {path_type} training data")
    generate_training_data(path_type, n_steps)
    '''

    # PREPROCESS
    out_type = 'q' # 'path' or 'q'
    data_file = 'all_lift.npy'
    
    n_steps, x, y, x_val, y_val, x_test, y_test, names_test = preprocess_data(data_file)
    chess_net = build_model(n_steps)
   
    # TRAIN
    history = chess_net.fit(x, y, epochs=5, validation_data=(x_val,y_val),)
    # plot loss during training
    plt.title('Loss vs Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # EVALUATE
    results = chess_net.evaluate(x_val, y_val)

    # TEST
    test_together(chess_net, x_test, y_test, n_steps, names_test, 4)
    time_test_together(chess_net, x_test, names_test, n_steps)
  
### Functions for Nested Network architecture ###
def build_pmodel(n_steps):
    "builds a model specific for predicting paths"
    dim = 3 * n_steps
    
    inputs = Input(shape=(6,))
    x = Rescaling(scale=1/500)(inputs)
    x = Dense(dim*25, activation='tanh')(x)
    #x = Dense(dim*25, activation='sigmoid')(x)
    x = Dense(dim)(x)
    outputs = Rescaling(scale=500)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="rmsprop", loss='mse', metrics=['accuracy'])
    model.summary()
       
    return model

def build_ikmodel():
    """builds a model specific to predicting inverse kinematics"""
    dim = 4
    
    inputs = Input(shape=(3,))
    x = Rescaling(scale=1/500)(inputs)
    x = Dense(dim*50, activation='tanh')(x)
    x = Dense(dim*50, activation='sigmoid')(x)
    outputs = Dense(dim, activation='linear')(x)
    

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="rmsprop", loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
       
    return model

def preprocess_data2(filename):
    """reads in the numpy arrays of dictionaries and produces training and validation data for one pathplanning net and another ik net"""
    all_moves = np.load(filename, allow_pickle=True)
    n_moves = len(all_moves)
    n_steps = np.size(all_moves[0]['path'],1)
    val_num = n_moves//4

    # for path net
    path_input = np.zeros((n_moves,6))
    path_output = np.zeros((n_moves,n_steps*3))
    for i_move in range(n_moves):
        path_input[i_move,:3] = all_moves[i_move]["start"]
        path_input[i_move,3:] = all_moves[i_move]["goal"]
        path_output[i_move,:] = all_moves[i_move]['path'].flatten(order='F')

    path_x = path_input[:-val_num,:]
    path_y = path_output[:-val_num,:]
    path_xval = path_input[-val_num:,:]
    path_yval = path_output[-val_num:,:]

    # for ik net
    ik_input = np.zeros((3,n_moves,n_steps))
    ik_output = np.zeros((4,n_moves,n_steps))
    for i_move in range(n_moves):
        for i_step in range(n_steps):
            ik_input[:,i_move,i_step] = all_moves[i_move]['path'][:,i_step]
            ik_output[:,i_move,i_step] = all_moves[i_move]['q'][:,i_step]
    ik_input = ik_input.reshape((3,n_moves*n_steps)).T
    ik_output = ik_output.reshape((4,n_moves*n_steps)).T

    ik_x = ik_input[:-val_num,:]
    ik_y = ik_output[:-val_num,:]
    ik_xval = ik_input[-val_num:,:]
    ik_yval = ik_output[-val_num:,:]
    
    return n_steps, path_x, path_y, path_xval, path_yval, ik_x, ik_y, ik_xval, ik_yval
     
def plot_paths(ref_path, nn_path, path_refpath_nnthetas, path_nnpath_nnthetas):
    paths = [ref_path, nn_path, path_refpath_nnthetas, path_nnpath_nnthetas]
    path_names = ['Reference Path', 'NN-generated Path', 'NN-generated Thetas from Ref Path', 'NN-generated Thetas from NN Path']
    fig = plt.figure()
    cm2 = ChessMoves()

    for i_path in range(4):
        ax = plt.subplot(2,2,i_path+1, projection='3d')
        plt.title(path_names[i_path])
        cm2.plot_board(ax)
        ax.scatter3D(paths[i_path][0],paths[i_path][1],paths[i_path][2])
    plt.show()

def plot_training(p_history, ik_history):
    # plot loss during training
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Error while Training Path')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.plot(p_history.history['loss'], label='train')
    plt.plot(p_history.history['val_loss'], label='test')
    plt.ylim([0,100])
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('Error while Training IK')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.plot(ik_history.history['loss'], label='train')
    plt.plot(ik_history.history['val_loss'], label='test')
    # plt.ylim([0,10])
    plt.legend()
    plt.show()

def test_separate(path_net, ik_net, path_xval, path_yval, test_idx):
    
    n_moves = np.size(path_yval,0)
    n_steps = int(np.size(path_yval,1)/3)
    test_idx = (n_moves % test_idx) - 1

    cm = ChessMoves()
    
    path_xtest = path_xval[test_idx,:].reshape((1,6))
    path_ytest = path_yval[test_idx,:].reshape((3,n_steps), order='F')
    
    # reference path
    theta_y = cm.inverse_kinematics(path_ytest)
    # print(f"Ref Path: {path_ytest} \nRef Thetas: {theta_y}")
    
    # neural net paths
    nn_path = path_net.predict(path_xtest).reshape((3,n_steps), order='F')
    refpath_nnthetas = np.zeros((4,n_steps))
    nnpath_nnthetas = np.zeros((4,n_steps))
    for step in range(n_steps):
        refpath_nnthetas[:,step] = ik_net.predict(path_ytest[:,step].reshape((1,3)))
        nnpath_nnthetas[:,step] = ik_net.predict(nn_path[:,step].reshape((1,3)))
    
    # test nn ik by doing forward kinematics to see what the path it would take
    path_refpath_nnthetas = cm.forward_kinematics(refpath_nnthetas)
    path_nnpath_nnthetas = cm.forward_kinematics(nnpath_nnthetas)

    plot_paths(path_ytest, nn_path, path_refpath_nnthetas, path_nnpath_nnthetas)
    # print(f"NN Path: {nn_path} \nNN-ref Path: {path_refpath_nnthetas} \nNN-nn Path: {path_nnpath_nnthetas} \nNN-ref Thetas: {refpath_nnthetas} \nNN Thetas: {nnpath_nnthetas}")
    cm.plot_robot(nnpath_nnthetas, path_ytest)

def time_test_separate(pmodel, ikmodel, x, n_steps):
    cm3 = ChessMoves()
    trad_times = np.zeros((3,1))
    nn_times = np.zeros((3,1))
    for j in range(3):
        start_trad = time.time()
        print('Traditionally solving 10 moves')
        for i in range(10):
            path = cm3.generate_flat_path(x[i,:3],x[i,3:],n_steps)
            thetas = cm3.inverse_kinematics(path)
        trad_time = time.time() - start_trad
        print(f"Traditional time: {trad_time}")
        trad_times[j] = trad_time

        print('Neural Net solving 10 moves')
        start_nn = time.time()
        for i in range(10):
            input = x[i,:].reshape((1,6))
            nn_path = pmodel.predict(input).reshape((3,n_steps), order='F')
            # nn_path = cm3.generate_flat_path(x[i,:3],x[i,3:],n_steps)
            nnthetas = np.zeros((4,n_steps))
            for step in range(n_steps):
                nnthetas[:,step] = ikmodel.predict(nn_path[:,step].reshape((1,3)))
        nn_time = time.time() - start_nn
        print(f"Neural Net time: {nn_time}")
        nn_times[j] = nn_time
    
    print(f'Average Traditional time: {np.mean(trad_times)}')
    print(f'Average Neural Net time: {np.mean(nn_times)}')
  
def main_separate():
    '''
    # GENERATE DATA
    path_type = 'lift'
    n_steps = 20
    print(f"generating {path_type} training data")
    generate_training_data(path_type, n_steps)
    '''
    # PREPROCESS
    data_file = 'all_lift.npy'
    n_steps, path_x, path_y, path_xval, path_yval, ik_x, ik_y, ik_xval, ik_yval = preprocess_data2(data_file)

    path_net = build_pmodel(n_steps)
    ik_net = build_ikmodel()
    
    # TRAIN
    p_history = path_net.fit(path_x, path_y, epochs=150, validation_data=(path_xval,path_yval),)
    ik_history = ik_net.fit(ik_x, ik_y, epochs=20, validation_data=(ik_xval,ik_yval),)
    plot_training(p_history, ik_history)

    # TEST
    test_separate(path_net, ik_net, path_xval, path_yval, 155)
    time_test_separate(path_net, ik_net, path_xval, n_steps)
    
if __name__ == '__main__':
    main_together()
    main_separate()