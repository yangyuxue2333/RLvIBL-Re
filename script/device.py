## ================================================================ ##
## MARKOV_DEVICE.PY                                                        ##
## ================================================================ ##
## A simple ACT-R device for the Gambling task                     ##
## -----------------------------------------                        ##
## This is a device that showcases the unique capacities of the new ##
## JSON-RPC-based ACT-R interface. The device is written in Python, ##
## and interacts with ACT-R entirely through Python code.           ##
## ================================================================ ##



import os
import time
from typing import Dict

import actr
import random
import pandas as pd
import os.path
PARAMETER_NAMES = ["ans", "bll", "lf", "egs", "alpha", "r"]

class GambleTask:
    def __init__(self, setup=False):
        self.response = None
        self.response_time = 0.0
        self.trials = []

        # init parameters
        self.subject_id = None
        self.reward = 0.0
        self.parameters = {}

        if setup:
            self.setup()

    def setup(self, model="model1",
              param_set=None,
              reload=True,
              subject_id=None,
              verbose=False):
        self.model = model
        self.verbose = verbose
        self.subject_id = subject_id

        # init working dir
        script_dir = os.path.join(os.path.dirname(os.path.realpath('../__file__')), 'script')

        self.add_actr_commands()

        if reload:
            # schedule event of detect production/reward before loading model
            # note: if not load, no need to schedule it again

            # load lisp model
            actr.load_act_r_model(os.path.join(script_dir, self.model + "_core.lisp"))
            self.set_parameters(param_set)
            actr.load_act_r_model(os.path.join(script_dir, self.model + "_body.lisp"))

        if subject_id == None:
            self.trials = self.create_stimuli()
        else:
            self.trials = GambleTask.load_stimuli(subject_dir='data/gambling_trials', subject_id=subject_id)

        window = actr.open_exp_window("GAMBLE TASK", width=500, height=250, visible=False)
        self.window = window
        actr.install_device(window)

        if verbose: print(self.__str__())

    def __str__(self):
        header = "######### SETUP MODEL " + self.model + " #########"
        parameter_info = ">> ACT-R PARAMETERS: " + str(self.parameters) + " <<"
        subject_info = ">> SUBJECT: [%s]" % (self.subject_id)
        return "%s\n \t%s\n \t%s\n" % (header, parameter_info, subject_info)

    def __repr__(self):
        return self.__str__()

    #################### PARAMETER SET ####################
    def get_parameters_name(self):
        if actr.current_model() == "MODEL1":
            param_names = ['ans', 'bll', 'lf']
        elif actr.current_model() == "MODEL2":
            param_names = ['alpha', 'egs', 'r']
        return param_names

    def get_parameter(self, param_name):
        """
        get parameter from current model
        :param keys: string, the parameter name (e.g. ans, bll, r1, r2)
        :return:
        """
        assert param_name in ("ans", "bll", "lf", "egs", "alpha", "r")
        if param_name == "r":
            return self.reward
        else:
            return actr.get_parameter_value(":" + param_name)

    def get_parameters(self, *kwargs):
        param_set = {}
        for param_name in kwargs:
            param_set[param_name] = self.get_parameter(param_name)
        return param_set

    def set_parameters(self, kwargs):
        """
        set parameter to current model
        :param kwargs: dict pair, indicating the parameter name and value (e.g. ans=0.1, r1=1, r2=-1)
        :return:
        """
        # global reward
        update_parameters = self.parameters.copy()

        if kwargs:
            update_parameters.update(kwargs)
            for key, value in kwargs.items():
                if key == "r":
                    self.reward = value
                else:
                    actr.set_parameter_value(':' + key, value)
            self.parameters = update_parameters
        else:
            self.parameters = self.get_default_parameters()

    def get_default_parameters(self):
        """
        default parameter sets
        """
        return self.get_parameters(*PARAMETER_NAMES)

    #################### TASK ####################

    def add_actr_commands(self):
        # monitor the output-key
        actr.add_command("paired-response", self.respond_to_key_press,
                         "Paired associate task key press response monitor")
        actr.monitor_command("output-key", "paired-response")

    def remove_actr_commands(self):
        actr.remove_command_monitor("output-key", "paired-response")
        actr.remove_command("paired-response")

    # def task(self, trials):
    #     """
    #     This function present task and monitor response from model
    #     :param size: number of trials to present
    #     :param trials: the trial list
    #     :return:
    #     """
    #     self.add_actr_commands()
    #     result = self.do_experiment(trials)
    #     self.remove_actr_commands()
    #
    #     return result

    def respond_to_key_press(self, model, key):
        """
        This function is set to monitor the output-key command, will be called whenever
        a key is pressed in the experiment window
        :param model: name of the model. if None, indicates a person interacting with
                    the window
        :param key: string name of the key
        :return:
        """
        # global response, response_time

        # record the latency of key-press (real time, not ACT-R time)
        self.response_time = actr.mp_time() - self.onset  #actr.get_time(model)
        self.response = key
        # print("TEST: in respond_to_key_press: ", self.response, self.response_time)

    def do_guess(self, prompt, window):
        """
        this function allows model to do first half of the experiment, guessing
        :param prompt:"?"
        :param window:
        :return: response "f" for less, or "j" for more
        """

        # display prompt
        actr.clear_exp_window(window)
        actr.add_text_to_exp_window(window, prompt, x=150, y=150)

        # wait for response
        # global response
        response = ''
        self.onset = actr.mp_time()

        # start = actr.get_time()
        actr.run_full_time(5)
        # time = self.response_time - start

    def do_feedback(self, feedback, window):
        """
        This  function allows the model to encode feedback
        :param feedback: "win" or "lose"
        :param window:
        :return:
        """

        actr.clear_exp_window(window)
        actr.add_text_to_exp_window(window, feedback, x=150, y=150)

        actr.run_full_time(5)

        # implement reward
        if actr.current_model() == "MODEL2":
            if feedback == "Reward":
                actr.trigger_reward(self.reward)
            elif feedback == "Punishment":
                actr.trigger_reward(-1.0 * self.reward)

    def do_experiment(self):
        """
        This function run the experiment, and return simulated model behavior
        :param size:
        :param trials:
        :param human:
        :return:
        """
        # actr.reset()

        result = []
        for trial in self.trials:
            prompt, feedback, block_type = trial

            # guess
            self.do_guess(prompt, self.window)

            # encode feedback
            self.do_feedback(feedback, self.window)

            # save simulation data
            result.append((feedback, block_type, self.response, self.response_time))

        return result

    def create_block(self, num_trials=8, num_reward=6, num_punish=1, num_neutral=1, block_type="MostlyReward", shuffle=False):
        """
        This function create experiment stimuli by blocks
        :param num_trials: number of trials =8
        :param num_reward: number of reward trials =6 (Mostly Reward  Block)
        :param num_punish: number of reward trials =1 (Mostly Reward  Block)
        :param num_neutral: number of reward trials =1 (Mostly Reward  Block)
        :param block_type: Mostly Reward  Block or Mostly Punishment  Block
        :param shuffle: whether to randomly shuffle trials within blocks
        :return: a block of trials (8)
        """
        prob_list = ["?"] * num_trials
        feedback_list = ["Reward"] * num_reward + ["Punishment"] * num_punish + ["Neutral"] * num_neutral
        block_list = [block_type] * num_trials
        trials = list(zip(prob_list, feedback_list, block_list))
        if shuffle: random.shuffle(trials)
        return trials

    def create_stimuli(self, num_run=1):
        trials = []
        MR1 = self.create_block(8, 6, 1, 1, "MostlyReward", True) + self.create_block(8, 4, 2, 2, "MostlyReward", True)
        MR2 = self.create_block(8, 6, 1, 1, "MostlyReward", True) + self.create_block(8, 4, 2, 2, "MostlyReward", True)
        MP1 = self.create_block(8, 1, 6, 1, "MostlyPunishment", True) + self.create_block(8, 2, 4, 2, "MostlyPunishment", True)
        MP2 = self.create_block(8, 1, 6, 1, "MostlyPunishment", True) + self.create_block(8, 2, 4, 2, "MostlyPunishment", True)
        r1_trials = MR1 + MP1 + MP2 + MR2
        r2_trials = MP1 + MR1 + MP2 + MR2

        if num_run == 1:
            trials = r1_trials
        elif num_run == 2:
            trials = r1_trials + r2_trials
        else:
            trials = None
        return trials

    @staticmethod
    def load_stimuli(subject_dir = 'data/gambling_trials', subject_id = '100307_fnca'):
        """
        This function enables the model to simulate based on specific HCP subj  stimuli order being accessed
        :param HCPID:
        :return:
        """
        parent_dir = os.path.dirname(os.getcwd())
        stim = pd.read_csv(os.path.join(parent_dir, subject_dir, subject_id + ".csv"), usecols=["TrialType", "BlockType"])
        stim["Probe"] = "?"
        stim = stim[['Probe', 'TrialType', 'BlockType']]
        trials = [tuple(x) for x in stim.to_numpy()]
        return trials

    def experiment(self):
        """
        This function call create_block() and task() to run experiment
        :param num_run: default =1, but could be 2
        :return: a dataframe of model outputs, with "TrialType", "BlockType", "Response", "RT" as columns
        """
        model_result = self.do_experiment()
        self.remove_actr_commands()

        model_result = pd.DataFrame(model_result, columns=["TrialType", "BlockType", "Response", "RT"])
        model_result["BlockTrial"] = list(range(0, 8)) * int(len(model_result) / 8)
        model_result["Trial"] = model_result.index
        return model_result

    @staticmethod
    def print_averaged_results(model_data):
        """
        This function print aggregated results group by trial type
        :param model_data: model output
        :return: aggregated results
        """
        print(model_data.groupby("TrialType").mean())
        print()
        print(model_data["TrialType"].value_counts(normalize=True))
        print()
        print(model_data.groupby("Response").mean())
        print()
        print(model_data["Response"].value_counts(normalize=True))
