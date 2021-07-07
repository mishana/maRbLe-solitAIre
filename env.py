from enum import IntEnum

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class CellState(IntEnum):
    EMPTY = 0
    FULL = 1
    DUMMY = 2


class MarbleAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class MarbleSolitaireEnv(gym.Env):
    """A Marble Solitaire Environment that represents a valid board with marbles"""

    BOARD_HEIGHT = 7
    BOARD_WIDTH = 7
    DUMMY_CELLS = 4 * 4  # marbles can't touch here

    MAX_MARBLES_NUM = BOARD_HEIGHT * BOARD_WIDTH - DUMMY_CELLS - 1
    MAX_ACTIONS_PER_MARBLE = len(MarbleAction)

    def __init__(self, init_fig=False, interactive_plot=False):
        super(MarbleSolitaireEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        # start from an (i,j) source pos, and move up/down/left/right (by eating.. nom nom)
        self.action_space = spaces.MultiDiscrete([self.BOARD_HEIGHT, self.BOARD_WIDTH, self.MAX_ACTIONS_PER_MARBLE])

        # (0) - no marble,
        # (1) - marble is placed in the cell,
        # (2) - a dummy cell
        self.observation_space = spaces.MultiDiscrete([len(CellState)] * self.BOARD_HEIGHT * self.BOARD_WIDTH)
        # self.observation_space = spaces.Box(low=CellState.EMPTY, high=CellState.DUMMY,
        #                                     shape=(self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)

        # additional init logic

        self.board = self._initial_board()  # TODO: is necessary?
        self.marbles_num = self.MAX_MARBLES_NUM
        self.step_counter = 0

        if init_fig:
            self._init_fig(interactive_plot)
        else:
            self.interactive_plot = False

    def _init_fig(self, interactive_plot=True):
        """
        Initializes the figure and axes for the rendering.
        Parameters
        ----------
        interactive_plot : bool (default True)
            Whether the plot is interactive or not.
        """
        if interactive_plot:
            plt.ion()
        self.fig = plt.figure(figsize=(10, 10))

    def _initial_board(self):
        board = np.full(fill_value=CellState.FULL, shape=(self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)

        board[self.BOARD_HEIGHT // 2, self.BOARD_WIDTH // 2] = CellState.EMPTY  # middle cell is empty
        board[:2, :2] = CellState.DUMMY
        board[-2:, :2] = CellState.DUMMY
        board[:2, -2:] = CellState.DUMMY
        board[-2:, -2:] = CellState.DUMMY

        return board

    def _is_valid_action(self, i, j, a):
        if self.board[i, j] != CellState.FULL:
            return False

        if a == MarbleAction.UP:
            return i >= 2 and self.board[i - 1, j] == CellState.FULL and self.board[i - 2, j] == CellState.EMPTY
        elif a == MarbleAction.DOWN:
            return i < self.BOARD_HEIGHT - 2 and self.board[i + 1, j] == CellState.FULL and self.board[
                i + 2, j] == CellState.EMPTY
        elif a == MarbleAction.LEFT:
            return j >= 2 and self.board[i, j - 1] == CellState.FULL and self.board[i, j - 2] == CellState.EMPTY
        else:  # MarbleAction.RIGHT
            return j < self.BOARD_WIDTH - 2 and self.board[i, j + 1] == CellState.FULL and self.board[
                i, j + 2] == CellState.EMPTY

    def _state_reward(self):
        return -self.marbles_num

    def _finish_reward(self):
        if self.marbles_num == 1:
            if self.board[self.BOARD_HEIGHT // 2, self.BOARD_WIDTH // 2] == CellState.FULL:
                return 0
            else:
                return -self.MAX_MARBLES_NUM
        else:
            return 0

    def _reward(self):
        return self._state_reward() + self._finish_reward() - self.step_counter

    def step(self, action):
        self.step_counter += 1

        i, j, a = action

        if self.board[i, j] == CellState.DUMMY:
            obs = self.board.flatten()
            reward = self._reward() * 100
            done = False
            info = {}

            return obs, reward, done, info

        # TODO: update reward/cost etc.
        # Execute one time step within the environment
        if not self._is_valid_action(i, j, a):
            obs = self.board.flatten()
            reward = self._reward() * 10
            done = False
            info = {}

            return obs, reward, done, info

        self.board[i, j] = CellState.EMPTY

        if a == MarbleAction.UP:
            self.board[i - 1, j] = CellState.EMPTY
            self.board[i - 2, j] = CellState.FULL
        elif a == MarbleAction.DOWN:
            self.board[i + 1, j] = CellState.EMPTY
            self.board[i + 2, j] = CellState.FULL
        elif a == MarbleAction.LEFT:
            self.board[i, j - 1] = CellState.EMPTY
            self.board[i, j - 2] = CellState.FULL
        else:  # MarbleAction.RIGHT
            self.board[i, j + 1] = CellState.EMPTY
            self.board[i, j + 2] = CellState.FULL

        self.marbles_num -= 1

        obs = self.board.flatten()
        reward = self._reward()
        done = self.marbles_num == 1 and self.board[self.BOARD_HEIGHT // 2, self.BOARD_WIDTH // 2] == CellState.FULL
        info = {}

        return obs, reward, done, info

    def reset(self):
        # TODO: init cost/reward, etc... and step counter?
        # Reset the state of the environment to an initial state
        self.board = self._initial_board()
        self.marbles_num = self.MAX_MARBLES_NUM
        self.step_counter = 0

        return self.board.flatten()

    def render(self, action=None, show_action=False, show_axes=True):
        """
        Renders the current state of the environment.
        Parameters
        ----------
        action : tuple of three ints or None (default None)
            If not None, a tuple (pos_i, pos_j, marble_action) of ints indicating the position on the board,
            and the current marble move direction - up/down/left/right.
        show_action : bool (default False)
            Indicates whether or not to change the color of the marble being moved and the marble being eaten in
            the rendering, in order to be able to visualise the action selected.
        show_axes : bool (default False)
            Whether to display the axes in the rendering or not.
        """
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(show_axes)
        ax.axes.get_yaxis().set_visible(show_axes)
        if show_action:
            assert action is not None
            y, x, a = action   # FLIP x, y here because of (i,j) and (x,y) "un-sync"

            if a == MarbleAction.UP:
                dx, dy = 0, 1
            elif a == MarbleAction.DOWN:
                dx, dy = 0, -1
            elif a == MarbleAction.RIGHT:
                dx, dy = 1, 0
            else:  # MarbleAction.LEFT
                dx, dy = -1, 0

            jumped_pos = (x + dx, y + dy)

            for pos, value in np.ndenumerate(self.board):
                pos = pos[::-1]   # FLIP pos here because of (i,j) and (x,y) "un-sync"

                if value == CellState.FULL:
                    if pos == (x, y):
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='brown', fill=True))
                    elif pos == jumped_pos:
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='black', fill=True))
                    else:
                        ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
                if value == CellState.EMPTY:
                    ax.add_patch(
                        matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))

        else:
            assert action is None
            for pos, value in np.ndenumerate(self.board):
                pos = pos[::-1]   # FLIP pos here because of (i,j) and (x,y) "un-sync"

                if value == CellState.FULL:
                    ax.add_patch(matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=True))
                if value == CellState.EMPTY:
                    ax.add_patch(
                        matplotlib.patches.Circle(xy=pos, radius=0.495, color='burlywood', fill=False, linewidth=1.5))

        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.axis('scaled')
        plt.title('Current State of the Board')
        self.fig.canvas.draw()
        # [p.remove() for p in reversed(ax.patches)]


if __name__ == '__main__':
    env = MarbleSolitaireEnv(init_fig=True, interactive_plot=True)
    env.render()
    plt.show()

    i, j, a = 3, 1, MarbleAction.RIGHT
    assert env._is_valid_action(i, j, a)
    env.render(action=(i, j, a), show_action=True)
    plt.show()
    env.step((i, j, a))
    env.render()
    plt.show()

    i, j, a = 3, 4, MarbleAction.LEFT
    assert env._is_valid_action(i, j, a)
    env.render(action=(i, j, a), show_action=True)
    plt.show()
    env.step((i, j, a))
    env.render()
    plt.show()

    # rng = np.random.default_rng()
    # for _ in range(1000):
    #     i, j, a = rng.integers(low=[0, 0, 0], high=[6, 6, 3], size=3)
    #     env.step((i, j, a))
    #
    #     if env._is_valid_action(i, j, a):
    #         env.render(action=(i, j, a), show_action=True)
    #         plt.show()
