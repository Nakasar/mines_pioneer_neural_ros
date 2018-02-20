from math import sin, cos, pi

class Movement_equations():
    def __init__(self, r, R, size):
        self.r = r
        self.R = R
        self.size = size

    def get_x_speed(self, q1, q2, theta):
        return self.r / 2 * (q1 + q2) * cos(theta)

    def get_y_speed(self, q1, q2, theta):
        return self.r / 2 * (q1 + q2) * sin(theta)

    def get_theta_speed(self, q1, q2):
        return self.r / (2 * self.R) * (q1 - q2)

    def get_delta_x(self, theta):
        return (self.r / 2) * cos(theta)

    def get_delta_y(self, theta):
        return (self.r / 2) * sin(theta)

    def get_delta_theta_q1(self):
        return (-1) * self.r / (2 * self.R)

    def get_delta_theta_q2(self):
        return self.r / (2 * self.R)

    def get_jx(self, x, delta_x, delta_t):
        return x * delta_t * delta_x

    def get_jy(self, y, delta_y, delta_t):
        return y * delta_t * delta_y

    def get_jtheta(self, theta, delta_theta, delta_t):
        return theta * delta_t * delta_theta

    def get_grad(self, jx, jy, jtheta, delta_t):
        return (-1) / (delta_t**2) * (jx/self.size + jy/self.size + jtheta/(2 * pi))

def get_grad(self, r, R, size, x, y, theta, x_target, y_target, theta_target, delta_t):
    equations = Movement_equations(r, R, size)

    relative_x = x - x_target
    relative_y = y - y_target
    relative_theta = (theta - theta_target + pi)%(2 * pi) - pi

    delta_x = equations.get_delta_x(theta)
    delta_y = equations.get_delta_y(theta)
    delta_theta_q1 = equations.get_delta_theta_q1()
    delta_theta_q2 = equations.get_delta_theta_q2()

    # The target and theta_shift are taken into account here (with relative variables)
    jx = equations.get_jx(relative_x, delta_x, delta_t)
    jy = equations.get_jy(relative_y, delta_y, delta_t)
    jtheta_q1 = equations.get_jtheta(relative_theta, delta_theta_q1, delta_t)
    jtheta_q2 = equations.get_jtheta(relative_theta, delta_theta_q2, delta_t)

    #jtheta_q1 = 0
    #jtheta_q2 = 0

    grad_1 = equations.get_grad(jx, jy, jtheta_q1, delta_t)
    grad_2 = equations.get_grad(jx, jy, jtheta_q2, delta_t)

    return [grad_1, grad_2]

