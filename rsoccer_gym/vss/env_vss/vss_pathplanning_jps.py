from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
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


import numpy as np
import random
import time 
import math
import gym


# Compute the distance betwewn two points (x,y)
def distance( point1: list, point2: list ) -> float:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def menor_angulo( vector1: list, vector2: list ):
    angle = math.acos( np.dot( vector1, vector2 ) )
    if np.cross( vector1, vector2 ) > 0:
        return -angle
    return angle

def close_to_x( x: float, range: float = 0.15 ):
    return np.clip( x + np.random.uniform( -range, range, 1)[0], -0.5, 0.5 )

def close_to_y( x: float, range: float = 0.15):
    return np.clip(x + np.random.uniform(-range, range, 1)[0], -0.5, 0.5)

def transform(v1, ang):
    mod = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v1 = (v1[0] / mod, v1[1] / mod)
    mn = menor_angulo(v1, (math.cos(ang), math.sin(ang)))
    return mn, (math.cos(mn) * mod, math.sin(mn) * mod)



""" This environment controls a singl e robot in a VSS soccer League 3v3 match
    Description:
        >>> .    
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
        >>> Sum of Rewards:
            Goal
            Ball Potential Gradient
            Move to Ball
            Energy Penalty
    Starting State:
        >>> Randomized Robots and Ball initial Position
    Episode Termination:
        >>> 5 minutes match time
"""
class vss_pathplanning_jps( VSSBaseEnv ):
    
    def __init__(self):
        # Construtor da classe VSSBaseEnv
        super().__init__( field_type = 0, n_robots_blue = 1, n_robots_yellow = 3, time_step = 0.025 )
        # Actions 
        self.action_space = gym.spaces.Box( 
            low   = -1, 
            high  =  1, 
            shape = (2,), 
            dtype = np.float32 
        )
        # Observations 
        self.observation_space = gym.spaces.Box( 
            low   = -self.NORM_BOUNDS, 
            high  = self.NORM_BOUNDS, 
            shape = (17,), 
            dtype = np.float32
        )
        # Initialize Class Atributes
        self.previous_ball_potential: float = None
        self.reward_shaping_total: dict     = None
        self.v_wheel_deadzone: float        = 0.05
        self.actions: dict                  = None
        
        self.ou_actions: list = []
        for _ in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(
                    self.action_space, 
                    dt = self.time_step
                )
            )
        self.plotting_data: list    = []
        self.difficulty: float      = 0.0  
        print("Environment initialized")


    # If the mean rewards gets to 450 points, it is maximum difficulty
    def set_diff(self, mean_rewards, max_mean_rewards = 450 ):
        diff = min( 1, max( 0.10, mean_rewards) / max_mean_rewards )
        if diff > 0.60 and self.difficulty == 0.1:
            self.difficulty = 0.25
        elif diff > 0.70 and self.difficulty == 0.25:
            self.difficulty = 0.55
        elif diff > 0.80 and self.difficulty == 0.55:
            self.difficulty = 1.00


    # Reseta o environment personalizado e chama o super.reset() para resetar o ambiente simulado 
    def reset(self):
        print( f"Env. difficulty: {self.difficulty}" )
        self.previous_ball_potential = None
        self.reward_shaping_total = None
        self.actions = None

        for ou in self.ou_actions:
            ou.reset()
        self.plotting_data.append([(0, 0)])

        return super().reset()
    

    # realiza um passo no ambiente simulado 
    def step( self, action ):
        if self.plotting_data[-1][-1] != ( self.frame.robots_blue[0].x, self.frame.robots_blue[0].y, ):
            self.plotting_data[-1].append( (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y) )
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total


    def _frame_to_observations(self):
        return self.observations_atacante()

    def observations_atacante(self):
        return observations(self)

    def _get_goalkeeper_vels( self, _debug: bool = False ) -> list:
        # Pegar a posição do Goleiro 
        gk: Robot = self.frame.robots_yellow[0]     # Obter o goleiro
        gk_pos: list = [ gk.x, gk.y ]               # Obter a posição do goleiro 
        gk_angle: float = math.radians(gk.theta)    # Calcular a orientação do goleiro em radianos
        gk_v_wheel: list = [ 0, 0 ]                 # Referencia de velocidade do goleiro
        
        # Pegar a posição da bola  
        ball: Ball = self.frame.ball                # Obter a bola
        ball_pos: list = [ ball.x, ball.y ]         # Obter a posição da bola

        # Lista do target para ser seguido
        target_pos: list = [ 0.65, ball.y ]

        # Cria um target proporcional ao eixo Y da bola 
        sin_t = np.sin( time.time()*0.5 )*np.cos( time.time()*2)*0.5
        target_pos[0] = 0.65
        if abs(sin_t) > 0.4:
            target_pos[1] = 0.4 if sin_t > 0 else -0.4
        else:
            target_pos[1] = sin_t

        # Pegar a distancia entre o goleiro e a bola  
        robot2ball_diff: list = [ target_pos[0] - gk_pos[0], target_pos[1] - gk_pos[1] ]            # return [ dx, dy ]
        robot2ball_mag: float = np.sqrt((robot2ball_diff[0]) ** 2 + (robot2ball_diff[1]) ** 2)      # return scalar 
        robot2ball_ang: float = np.arctan2( robot2ball_diff[1], robot2ball_diff[0])                 # return scalar entre [ -pi e pi ]

        # Calcula a diferença entre o angulo do robo e o angulo gerado entre o robo e a bola 
        robot2ball_ang = np.degrees( robot2ball_ang) 
        gk_angle = np.degrees(gk_angle)
        
        diff_ang = (gk_angle - robot2ball_ang) % 360
        if _debug:
            print( f"r2b_Dang: {robot2ball_ang:.4f}, gk_Dang:{gk_angle:.4f}, diff_Dang:{diff_ang:.4f}, gk_Rang:{np.cos( np.radians(diff_ang)):.4f}, target_pos: [{target_pos[0]:.4f},{target_pos[1]:.4f}], gk_pos:{gk_pos[1]:.4f},{gk_pos[1]:.4f}]" )

        # Calcula a diferença de velocidade baseado no sinal de diff_ang 
        OFFSET_DIFF_ANGLE = 180
        if robot2ball_ang > 0:
            # if abs(diff_ang) > OFFSET_DIFF_ANGLE:
            #     # Manipula a roda Direita e Esquerda  
            #     gk_v_wheel = [ 1-np.cos( np.radians(diff_ang+OFFSET_DIFF_ANGLE)), np.cos( np.radians(diff_ang)) ]
            # else: 
            # Manipula somente a roda direita 
            gk_v_wheel = [ 1, np.cos( np.radians(diff_ang)) ]
        
        elif robot2ball_ang < 0:
            # if abs(diff_ang) > OFFSET_DIFF_ANGLE:
            #     # Manipula a roda esquerda e direita   
            #     gk_v_wheel = [ np.cos( np.radians(diff_ang)), 1-np.cos( np.radians(diff_ang-OFFSET_DIFF_ANGLE))]
            # else:
            # Manipula somente a roda esquerda  
            gk_v_wheel = [ np.cos( np.radians(diff_ang)), 1 ]
        else: 
            # Anda reto com velocidade máxima 
            gk_v_wheel = [ 1, 1 ]


        # Ajustar a velocidade máxima conforme necessário, saida de [-1, 1]
        max_speed = (robot2ball_mag + 0.4 )*30
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
        print( commands )

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

        # pos_frame.ball = Ball(x=pos[0], y=pos[1])
        pos_frame.ball = Ball( x = random.random()*0.6,  y = random.random()*0.6 )

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
