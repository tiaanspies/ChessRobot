"""
Class of n-link arm in 3D
Author: Takayuki Murooka (takayuki5168)
Edited by Noah Jones for this specific use case
"""
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Link:
    def __init__(self, dh_params):
        self.dh_params_ = dh_params

    def transformation_matrix(self):
        theta = self.dh_params_[0]
        alpha = self.dh_params_[1]
        a = self.dh_params_[2]
        d = self.dh_params_[3]

        st = math.sin(theta)
        ct = math.cos(theta)
        sa = math.sin(alpha)
        ca = math.cos(alpha)
        trans = np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d],
                          [0, 0, 0, 1]])

        return trans

    @staticmethod
    def basic_jacobian(trans_prev, ee_pos):
        pos_prev = np.array(
            [trans_prev[0, 3], trans_prev[1, 3], trans_prev[2, 3]])
        z_axis_prev = np.array(
            [trans_prev[0, 2], trans_prev[1, 2], trans_prev[2, 2]])

        basic_jacobian = np.hstack(
            (np.cross(z_axis_prev, ee_pos - pos_prev), z_axis_prev))
        return basic_jacobian


class NLinkArm:
    def __init__(self, dh_params_list):
        self.link_list = [Link(dh_params) for dh_params in dh_params_list]

    def transformation_matrix(self):
        trans = np.identity(4)
        for link in self.link_list:
            trans = np.dot(trans, link.transformation_matrix())
        return trans

    def forward_kinematics(self, trans, plot=False):
        

        x = trans[0, 3]
        y = trans[1, 3]
        z = trans[2, 3]
        alpha, beta, gamma = self.euler_angle(trans)

        if plot:
            self.plot_arm()

        return [x, y, z, alpha, beta, gamma]

    def basic_jacobian(self, trans):
        ee_pos = self.forward_kinematics(trans)[0:3]
        basic_jacobian_mat = []

        trans = np.identity(4)
        for link in self.link_list:
            basic_jacobian_mat.append(link.basic_jacobian(trans, ee_pos))
            trans = np.dot(trans, link.transformation_matrix())

        return np.array(basic_jacobian_mat).T

    def inverse_kinematics(self, ref_ee_pose, plot=False):
        for it in range(501):
            trans = self.transformation_matrix()
            ee_pose = self.forward_kinematics(trans)
            diff_pose = ref_ee_pose - ee_pose

            basic_jacobian_mat = self.basic_jacobian(trans)                
            
            alpha, beta, gamma = self.euler_angle(trans)

            K_zyz = np.array(
                [[0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
                 [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
                 [1, 0, math.cos(beta)]])
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz

            theta_dot = np.dot(
                np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
                np.array(diff_pose))
            self.update_joint_angles(theta_dot / 100.)

            # if (it) % 10 == 0:
            #     print(f"ITERATION {it}:{np.linalg.norm(diff_pose)}")

        if plot:
            self.plot_arm(ref_ee_pose)

        return self.get_joint_angles()
    
    def inverse_kinematics_grad_descent(self, ref_ee_pose, plot=False):
        def objective(joint_angles):
            self.set_joint_angles(joint_angles)
            trans = self.transformation_matrix()
            ee_pos = self.forward_kinematics(trans)[0:3]
            
            return np.linalg.norm(ref_ee_pose - np.array(ee_pos))

        initial_guess = self.get_joint_angles()
        result = minimize(objective, initial_guess, method='BFGS', options={'gtol': 0.1})
        self.set_joint_angles(result.x)
        
        if plot:
            self.plot_arm(ref_ee_pose)

        return self.get_joint_angles()

    def euler_angle(self, trans):

        alpha = math.atan2(trans[1][2], trans[0][2])
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) + math.pi
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) - math.pi
        beta = math.atan2(
            trans[0][2] * math.cos(alpha) + trans[1][2] * math.sin(alpha),
            trans[2][2])
        gamma = math.atan2(
            -trans[0][0] * math.sin(alpha) + trans[1][0] * math.cos(alpha),
            -trans[0][1] * math.sin(alpha) + trans[1][1] * math.cos(alpha))

        return alpha, beta, gamma

    def set_joint_angles(self, joint_angle_list):
        for i, angle in enumerate(joint_angle_list):
            self.link_list[i].dh_params_[0] = angle

    def update_joint_angles(self, diff_joint_angle_list):
        for i, diff in enumerate(diff_joint_angle_list):
            self.link_list[i].dh_params_[0] += diff

    def get_joint_angles(self):
        return [link.dh_params_[0] for link in self.link_list]

    def get_vertices(self):
        verts = np.zeros((3, len(self.link_list) + 1))
        trans = np.identity(4)

        for i, link in enumerate(self.link_list):
            trans = np.dot(trans, link.transformation_matrix())
            verts[:, i + 1] = trans[:3, 3]

        return verts

    def plot_arm(self, ref_ee_pose=None):
        fig = plt.figure()
        ax = Axes3D(fig)

        x_list = []
        y_list = []
        z_list = []

        trans = np.identity(4)

        x_list.append(trans[0, 3])
        y_list.append(trans[1, 3])
        z_list.append(trans[2, 3])
        for link in self.link_list:
            trans = np.dot(trans, link.transformation_matrix())
            x_list.append(trans[0, 3])
            y_list.append(trans[1, 3])
            z_list.append(trans[2, 3])

        ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4, mew=0.5)
        ax.plot([0], [0], [0], "o")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(-200, 200)
        ax.set_ylim(0, 500)
        ax.set_zlim(0, 300)

        if ref_ee_pose:
            ax.plot([ref_ee_pose[0]], [ref_ee_pose[1]], [ref_ee_pose[2]], "o")

        plt.show()
