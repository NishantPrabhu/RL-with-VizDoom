
""" 
Main script and trainer definition
"""

import os
import wandb
import common
import random
import argparse
import numpy as np
import vizdoom as vzd
import itertools as it
from time import sleep, time
from collections import deque
from datetime import datetime as dt
import skimage.transform as transform

import torch
import torch.nn as nn
import torch.optim as optim
import agents


COLORS = {
    "red": "\033[31m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[94m",
    "green": "\x1b[32m",
    "end": "\033[0m"
}

AGENTS = {
    'dqn': agents.DQNAgent,
    'double_dqn': agents.DoubleDQNAgent,
    'duel_dqn': agents.DuelDQNAgent
}

NUM_FRAMES = 1

class Trainer:

    def __init__(self, args):
        assert (args['agent'] in AGENTS.keys()), f"Invalid agent '{args['agent']}'; choose from {list(AGENTS.keys())}"
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)
        self.res_x, self.res_y = self.config['game']['resolution']
        self.frame_repeat = self.config['agent']['frame_repeat']
        self.batch_size = self.config['agent']['batch_size']
        self.args = args
        
        # Initialize game and logging
        self.initialize_game()
        self.agent = AGENTS[args['agent']](
            {**self.config['agent'], 'epochs': self.config['epochs']}, self.action_size, self.device
        )
        
        # Wandb
        run = wandb.init('vizdoom-playground')
        self.logger.write(f"Wandb: {run.get_url()}", mode='info')
        self.best_val_reward = -np.inf

        if args['load'] is not None:
            state = torch.load(os.path.join(args['load'], 'best_model.ckpt'))
            self.done_epochs = state['epoch']
            self.agent = state['agent']
            self.logger.record(f"Successfully loaded model from {args['load']}!", mode='info')

    def initialize_game(self):
        self.game = vzd.DoomGame()
        self.game.load_config(self.config['game']['config_file_path'])
        self.game.set_window_visible(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.init()

        # Define action space
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.action_size = len(self.actions)

    def preprocess_img(self, img):
        img = transform.resize(img, tuple([self.res_x, self.res_y]))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def save_best_state(self, epoch):
        state = {'epoch': epoch+1, 'agent': self.agent}
        torch.save(state, os.path.join(self.output_dir, 'best_model.ckpt'))

    def save_state(self, epoch):
        state = {'epoch': epoch+1, 'agent': self.agent}
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt')) 

    def validate(self, epoch):
        val_scores = []
        self.logger.record("{}/{}".format(epoch+1, self.config['epochs']), mode='val')
        self.agent.model.eval()

        for step in range(self.config['val_steps_per_epoch']):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = [self.preprocess_img(self.game.get_state().screen_buffer) for _ in range(NUM_FRAMES)]
                state = np.concatenate(state, axis=0)
                with torch.no_grad():
                    best_action_idx = self.agent.get_action(state)
                self.game.make_action(self.actions[best_action_idx], self.frame_repeat)
            
            r = self.game.get_total_reward()
            val_scores.append(r)
            common.progress_bar(progress=(step+1)/self.config['val_steps_per_epoch'], status=f"[Reward] {np.mean(val_scores)}") 

        common.progress_bar(progress=1.0, status=f"[Reward] {np.mean(val_scores)}")
        wandb.log({'Validation reward': np.mean(val_scores), 'Epoch': epoch+1})
        self.logger.write(f"[Reward] {np.mean(val_scores)}", mode='val')
        return np.mean(val_scores)

    def watch_performance(self):
        self.game.close()
        self.game.set_window_visible(True)
        self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.game.init()
        self.agent.model.eval()
        self.logger.record("Beginning preview!", mode='info')

        for epoch in range(self.config['episodes_to_watch']):
            self.game.new_episode()

            while not self.game.is_episode_finished():
                state = [self.preprocess_img(self.game.get_state().screen_buffer) for _ in range(NUM_FRAMES)]
                state = np.concatenate(state, axis=0)
                with torch.no_grad():
                    best_action_idx = self.agent.get_action(state)

                # The alternative below makes animations smooth
                self.game.set_action(self.actions[best_action_idx])
                for _ in range(self.frame_repeat):
                    self.game.advance_action()
            
            sleep(1.0)
            score = self.game.get_total_reward()
            self.logger.record("[Episode {}] Total reward: {:.4f}".format(epoch+1, score), mode='info')

    def train(self):
        self.logger.record("Beginning training!", mode='info')
        
        for epoch in range(self.config['epochs']):
            self.loss_meter = common.AverageMeter()
            self.reward_meter = common.AverageMeter()
            self.logger.record("{}/{}".format(epoch+1, self.config['epochs']), mode='train')
            self.game.new_episode()
            self.agent.model.train()
            completed_episodes = 0
            
            for step in range(self.config['train_steps_per_epoch']):
                state = [self.preprocess_img(self.game.get_state().screen_buffer) for _ in range(NUM_FRAMES)]
                state = np.concatenate(state, axis=0)                                              
                action = self.agent.get_action(state)                                                      
                reward = self.game.make_action(self.actions[action], self.frame_repeat)
                done = int(self.game.is_episode_finished())

                if not done:
                    next_state = [self.preprocess_img(self.game.get_state().screen_buffer) for _ in range(NUM_FRAMES)]
                    next_state = np.concatenate(next_state, axis=0)
                else:
                    next_state = np.zeros((NUM_FRAMES, self.res_x, self.res_y)).astype(np.float32)

                self.agent.replay_memory.add_transaction(
                    torch.from_numpy(state).float(), action, reward, torch.from_numpy(next_state).float(), done
                )
                # Learn from replay memory when enough samples have been generated
                if (step > self.batch_size):
                    loss_metrics = self.agent.learn_from_memory()
                    self.loss_meter.add(loss_metrics)
                    wandb.log({"Train loss": loss_metrics['Loss']})
                
                if done:
                    completed_episodes += 1
                    wandb.log({"Reward": self.game.get_total_reward()})
                    self.reward_meter.add({'Reward': self.game.get_total_reward()})
                    self.game.new_episode()

                common.progress_bar(
                    progress=(step+1)/self.config['train_steps_per_epoch'], 
                    status=self.loss_meter.return_msg() + self.reward_meter.return_msg())

            # Update target model
            if args['agent'] != 'dqn':
                self.agent.target_model.load_state_dict(self.agent.model.state_dict())

            # Logging
            common.progress_bar(progress=1.0, status=self.loss_meter.return_msg() + self.reward_meter.return_msg())
            wandb.log({'Average Train reward': self.reward_meter.return_metrics()['Reward'], 'Epoch': epoch+1})
            wandb.log({'Completed episodes': completed_episodes, 'Epoch': epoch+1})
            self.save_state(epoch)
            self.logger.write(self.reward_meter.return_msg() + f"[Completed episodes] {completed_episodes}", mode='train')

            # Validate if necessary
            if (epoch+1) % self.config['eval_every'] == 0:
                val_reward = self.validate(epoch)
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_best_state(epoch)

        self.logger.record("Training complete!", mode='info')
        self.watch_performance()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", "-t", type=str, required=True, help='Task to perform, train or watch')
    ap.add_argument("--config", "-c", type=str, required=True, help='Path to configuration file')
    ap.add_argument("--output", "-o", type=str, default=dt.now().strftime("%H-%M_%d-%m-%Y"), help='Name of output folder')
    ap.add_argument("--agent", "-a", type=str, default='dqn', help='Type of agent to be trained/used')
    ap.add_argument("--load", "-l", type=str, help='Path to directory containing best_model.ckpt to be loaded')
    args = vars(ap.parse_args())

    # Initialize trainer
    trainer = Trainer(args)
    
    if args['task'] == 'train':
        trainer.train()
    
    elif args['task'] == 'watch':
        if args['load'] is None:
            print(f"\n{COLORS['red']}[WARN] No model has been loaded! Performance might not be good!{COLORS['end']}\n")
        trainer.watch_performance()

    else:
        raise ValueError(f"Invalid task '{args['task']}'; choose from [train, watch]")