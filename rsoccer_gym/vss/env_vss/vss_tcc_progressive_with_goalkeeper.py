from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.vss.env_vss.shared_tcc import (
    observations,
    w_ball_grad_tcc,
    w_energy_tcc,
    w_move_tcc,
    goal_reward_tcc,
)

from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import numpy as np
import pickle
import random
import time 
import math
import gym


def distancia( o1, o2 ):
    return np.sqrt((o1.x - o2.x) ** 2 + (o1.y - o2.y) ** 2)

def close_to_x( x, range = 0.15 ):
    return np.clip(x + np.random.uniform(-range, range, 1)[0], -0.5, 0.5)

def close_to_y(x, range=0.15):
    return np.clip(x + np.random.uniform(-range, range, 1)[0], -0.5, 0.5)

def menor_angulo(v1, v2):
    angle = math.acos(np.dot(v1, v2))
    if np.cross(v1, v2) > 0:
        return -angle
    return angle

def transform(v1, ang):
    mod = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v1 = (v1[0] / mod, v1[1] / mod)
    mn = menor_angulo(v1, (math.cos(ang), math.sin(ang)))
    return mn, (math.cos(mn) * mod, math.sin(mn) * mod)




class vss_tcc_progressive_with_goalkeeper(VSSBaseEnv):
    """This environment controls a singl
    e robot in a VSS soccer League 3v3 match

        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    # go_to_point: list = [ ]
    point1 = [ random.random()*( -1 if random.randint(0,1) == 1 else 1), random.random()*( -1 if random.randint(0,1) == 1 else 1) ]         
    point2 = [ random.random()*( -1 if random.randint(0,1) == 1 else 1), random.random()*( -1 if random.randint(0,1) == 1 else 1) ]         
    point3 = [ random.random()*( -1 if random.randint(0,1) == 1 else 1), random.random()*( -1 if random.randint(0,1) == 1 else 1) ]         

    t1 = 0.0

    def __init__(self):
        super().__init__(
            field_type=0, n_robots_blue=1, n_robots_yellow=3, time_step=0.025
        )

        self.action_space = gym.spaces.Box( low = -1, high = 1, shape = (2,), dtype = np.float32 )
        
        self.observation_space = gym.spaces.Box(
            low = -self.NORM_BOUNDS, high = self.NORM_BOUNDS, shape = (17,), dtype = np.float32
        )

        # O que isso faz ?
        self.goleiro_teve_bola = False

        # Initialize Class Atributes
        self.previous_ball_potential = None

        self.reward_shaping_total = None
        self.v_wheel_deadzone: float = 0.05
        self.actions:dict = None

        self.ou_actions = []
        for _ in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )
        self.difficulty = 0.0  # starts easy
        self.plotting_data = []
        print("Environment initialized")


    def set_diff(self, mean_rewards, max_mean_rewards=450):
        diff = min(
            1, max(0.1, mean_rewards) / max_mean_rewards
        )  # if the mean rewards gets to 450 points, it is maximum difficulty

        if diff > 0.6 and self.difficulty == 0.1:
            self.difficulty = 0.25
        elif diff > 0.7 and self.difficulty == 0.25:
            self.difficulty = 0.55
        elif diff > 0.8 and self.difficulty == 0.55:
            self.difficulty = 1

    def reset(self):
        print(f"Env. difficulty: {self.difficulty}")
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None

        for ou in self.ou_actions:
            ou.reset()

        self.plotting_data.append([(0, 0)])

        # Falta colocar o goleiro amarelo na reta do gol e controlar ele a partir dessa posição
        self.t1 = time.time( )
        self.next_point()

        return super().reset()
    

    def next_point( self ):
        self.target = [ 0.6*random.random()*( -1 if random.randint(0,1) == 1 else 1), 0.6*random.random()*( -1 if random.randint(0,1) == 1 else 1) ]         
    
    def step(self, action):
        if self.plotting_data[-1][-1] != (
            self.frame.robots_blue[0].x,
            self.frame.robots_blue[0].y,
        ):
            self.plotting_data[-1].append(
                (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
            )
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):
        return self.observations_atacante()

    def observations_atacante(self):
        return observations(self)

    def _get_goalkeeper_vels( self ) -> list:
        # Pegar a posição do Goleiro 
        gk: Robot = self.frame.robots_yellow[0]     # Obter o goleiro
        gk_pos: list = [ gk.x, gk.y ]               # Obter a posição do goleiro 
        gk_angle: float = math.radians(gk.theta)    # Calcular a orientação do goleiro em radianos
        gk_v_wheel: list = [ 0, 0 ]                 # Referencia de velocidade do goleiro

        
        # Pegar a posição da bola  
        ball: Ball = self.frame.ball                # Obter a bola
        ball_pos: list = [ ball.x, ball.y ]         # Obter a posição da bola

        # Cria um target para ir até lá
        # ball_pos: list = [ 0.6, 0.50 ]        
        if time.time() - self.t1 > 2:
            ball_pos = self.target  
            if np.sqrt((gk_pos[0] - ball_pos[0]) ** 2 + (gk_pos[1] - ball_pos[1]) ** 2) < 0.05:
                self.next_point()
        
        # Pegar a distancia entre o goleiro e a bola  
        robot2ball_diff: list = [ ball_pos[0] - gk_pos[0], ball_pos[1] - gk_pos[1] ]            # return [ dx, dy ]
        robot2ball_mag: float = np.sqrt((robot2ball_diff[0]) ** 2 + (robot2ball_diff[1]) ** 2)  # return scalar 
        robot2ball_ang: float = np.arctan2( robot2ball_diff[1], robot2ball_diff[0])             # return scalar between [-pi, pi ]

        # Calcula a diferença entre o angulo do robo e o angulo gerado entre o robo e a bola 
        diff_ang = gk_angle - robot2ball_ang 
        print( gk_pos, robot2ball_diff)

        # Calcula a diferença de velocidade baseado no sinal de diff_ang 
        if diff_ang > 0:
            # Manipula a roda direita 
            gk_v_wheel = [ 1, np.cos( diff_ang) ]
        elif diff_ang < 0:
            # Manipula a roda esquerda  
            gk_v_wheel = [ np.cos( diff_ang), 1 ]
        else: 
            # Anda reto com velocidade máxima 
            gk_v_wheel = [ 1, 1 ]

        # Ajustar a velocidade máxima conforme necessário, saida de [-1, 1]
        max_speed = robot2ball_mag*20
        # print( gk_v_wheel )

        return [ max_speed * gk_v for gk_v in gk_v_wheel ]
        # return [ 255, 100 ]
    
    

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions[:2]
        gk_v_wheel_0, gk_v_wheel_1 = self._get_goalkeeper_vels()

        goalkeeper_move = Robot(
            yellow=True,
            id=0,
            v_wheel0=gk_v_wheel_0,
            v_wheel1=gk_v_wheel_1,
        )

        commands.append(goalkeeper_move)

        if (
            self.difficulty > 0.5
        ):  # if agent is 50% good, start slowly making other robots move in a random way
            movement = (self.difficulty - 0.2) / 0.8

            # Skip robot with id 0 which is the goalkeeper
            for i in range(1, self.n_robots_yellow):
                actions = self.ou_actions[self.n_robots_blue + i].sample()
                v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
                commands.append(
                    Robot(
                        yellow=True,
                        id=i,
                        v_wheel0=v_wheel0 * movement,
                        v_wheel1=v_wheel1 * movement,
                    )
                )

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = w_move_tcc
        w_ball_grad = w_ball_grad_tcc
        w_energy = w_energy_tcc
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "move": 0,
                "energy": 0,
                "ball_gradient": 0,
            }

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            reward = goal_reward_tcc
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            reward = -100
            goal = True
        else:

            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = w_ball_grad * self.__ball_grad()
                # Calculate Move ball
                move_reward = w_move * self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = w_energy * self.__energy_penalty()

                reward = grad_ball_potential + move_reward + energy_penalty

                self.reward_shaping_total["move"] += move_reward
                self.reward_shaping_total["energy"] += energy_penalty
                self.reward_shaping_total["ball_gradient"] += grad_ball_potential

        return reward, goal

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""

        def x(x: float = 0):
            return close_to_x(x, self.difficulty)

        def y(y: float = 0):
            return close_to_y(y, self.difficulty)

        def theta():
            return random.uniform(0, 360)

        pos_frame = Frame()

        places = KDTree()

        pos = (x(), y())
        places.insert(pos)

        pos_frame.ball = Ball(x=pos[0], y=pos[1])

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(-0.5), y())
        places.insert(pos)

        # posicao do agente
        pos_frame.robots_blue[0] = Robot(x=pos[0], y=pos[1], theta=theta())

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(0.6), close_to_y(0, 0.05))
        places.insert(pos)

        # posicao inicial do goleiro
        pos_frame.robots_yellow[0] = Robot(x=pos[0], y=pos[1], theta=theta())

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(), y(-0.4))
        places.insert(pos)

        pos_frame.robots_yellow[1] = Robot(x=pos[0], y=pos[1], theta=theta())

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(), y(0.4))
        places.insert(pos)

        pos_frame.robots_yellow[2] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        # recebe os valores da rede e converte (velocidade linear, velocidade angular)
        # para velocidades da roda entre -1 e 1

        # espaçamento entre rodas do carrinho, 1 para que o valor maximo seja 1 tbm
        L = 1

        vleft = (actions[0] - (actions[1] * L) / 2) * 1

        vright = (actions[0] + (actions[1] * L) / 2) * 1

        left_wheel_speed = vleft * self.max_v
        right_wheel_speed = vright * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def __ball_grad(self):
        """Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        """
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = math.sqrt(dx_d**2 + 2 * dy**2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step, -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __move_reward(self):
        """Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty
