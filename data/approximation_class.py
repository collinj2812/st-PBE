import time

import numpy as np
import do_mpc
import sys
import os
import pickle
import json
import casadi as ca
import torch
import torch.utils.data
# import keras
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
# import torch.optim.lr_scheduler
import multiprocessing as mp

import cryst

sys.path.append('./MSMPR')
sys.path.append('./Tubular')
sys.path.append('./Config')

import MSMPR_model
import tubular_model
import model_class
import config_data


# classes for torch model
class torch_model(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim1, hidden_dim2):
        super(torch_model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_dim1)
        # self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim2, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.1)
        # self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        # x= self.dropout(x)
        # x = self.sigmoid(self.fc2(x))
        # x = self.sigmoid(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc5(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Approximate:
    def __init__(self):
        self.last_element_only = None
        self.scheduler = None
        self.t_step = None
        self.no_discr = None
        self.method = None
        self.l = None
        self.best_model_weights = None
        self.val_losses = None
        self.train_losses = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.optimizer = None
        self.criterion = None
        self.learning_rate = None
        self.hidden_dim = None
        self.torch_model = None
        self.batch_size = None
        self.Y_val = None
        self.X_val = None
        self.Y_train = None
        self.X_train = None
        self.pca = None
        self.X_preprocessed = None
        self.Y_scaled = None
        self.X_scaled = None
        self.scaler_Y = None
        self.scaler_X = None
        self.Y = None
        self.X = None
        self.n_inputs = None
        self.n_states = None
        self.inputs = None
        self.states = None
        self.input_names = None
        self.state_names = None

    def load_data(self, simulation_obj_list, meta_data_list, reduce_PBE_state=True, last_element_only=False):
        # simulation_obj is from do-mpc
        # if reduce_PBE_state is True, only the mean diameter and width are used as output
        self.last_element_only = last_element_only

        # initialize lists of data
        self.states = []
        self.inputs = []

        # go through list of simulation_obj and meta_data

        # extract meta data from first simulation object
        self.method = meta_data_list[0]['PBE_method']
        self.t_step = simulation_obj_list[0]['simulator']['_time'][
            1]  # starts at 0, therefore first time step gives t_step
        self.no_discr = int(
            simulation_obj_list[0]['simulator']['_aux', 'mu'].shape[1] / 6)  # number of stages or discrete elements
        self.state_names = simulation_obj_list[0]['simulator'].model['_x'].keys()
        self.input_names = simulation_obj_list[0]['simulator'].model['_u'].keys()[1:]  # first name is default
        self.n_states = simulation_obj_list[0]['simulator'].model['n_x']
        self.n_inputs = simulation_obj_list[0]['simulator'].model['n_u']

        for i, sim_obj_i in enumerate(simulation_obj_list):
            if reduce_PBE_state:
                if self.method == 'SMOM' or self.method == 'QMOM':
                    mean, width = mean_width_moments(sim_obj_i['simulator']['_x', 'PBE_state'])
                    if last_element_only:
                        mean, width = mean[:, -1].reshape(-1, 1), width[:, -1].reshape(-1, 1)
                    # current_state = np.concatenate([mean, width], axis=1)
                    current_state = np.concatenate([mean], axis=1)  # only mean
                    # concatenate with all states besides PBE_state
                    for state_name in self.state_names:
                        if state_name != 'PBE_state':
                            if last_element_only:
                                current_state = np.concatenate(
                                    [current_state, sim_obj_i['simulator']['_x', state_name][:, -1].reshape(-1, 1)],
                                    axis=1)
                            else:
                                current_state = np.concatenate(
                                    [current_state, sim_obj_i['simulator']['_x', state_name]],
                                    axis=1)

                    # add maximum norm of relative supersaturation
                    rel_sup = sim_obj_i['simulator']['_aux', 'rel_S']
                    current_state = np.concatenate(
                        [current_state, rel_sup], axis=1)

                elif self.method == 'DPBE':
                    median = True  # median or mean
                    if median:
                        no_disc_PBE = int(sim_obj_i['simulator']['_x', 'PBE_state'].shape[1] / self.no_discr)
                        mean, width = mean_width_DPBE(sim_obj_i['simulator']['_x', 'PBE_state'], self.no_discr,
                                                      no_disc_PBE, meta_data_list[i]['L_i'])
                    else:
                        mean, width = mean_width_moments(sim_obj_i['simulator']['_aux', 'mu'])
                    if last_element_only:
                        mean, width = mean[:, -1].reshape(-1, 1), width[:, -1].reshape(-1, 1)
                    current_state = np.concatenate([mean, width], axis=1)
                    # current_state = np.concatenate([mean], axis=1)
                    for state_name in self.state_names:
                        if state_name != 'PBE_state':
                            if last_element_only:
                                current_state = np.concatenate(
                                    [current_state, sim_obj_i['simulator']['_x', state_name][:, -1].reshape(-1, 1)],
                                    axis=1)
                            else:
                                current_state = np.concatenate(
                                    [current_state, sim_obj_i['simulator']['_x', state_name]],
                                    axis=1)


                elif self.method == 'OCFE':
                    print('Convert OCFE to mean and width not implemented yet')
            else:
                current_state = sim_obj_i['simulator']['_x']
            self.inputs.append(sim_obj_i['simulator']['_u'])
            self.states.append(current_state)

    def setup_data(self, l=1, n_components=10):
        # setup of data matrices X and Y for Training, scaling and PCA
        # X: input data, Y: output data
        self.l = l
        self.X, self.Y = narx_data(self.states, self.inputs, l=self.l)

        # scale
        self.scaler_X, self.scaler_Y, self.X_scaled, self.Y_scaled = scale_data(self.X, self.Y)

        # pca
        self.pca, self.X_preprocessed = PCA_data(self.X_scaled, n_components=n_components)

        if n_components < 1:
            print(
                f'{self.pca.n_components_} components explain {np.sum(self.pca.explained_variance_ratio_)} of the variance')
            print(
                f'Compression factor is {(1 - self.pca.n_components_ / self.X_scaled.shape[1]) * 100}%. From {self.X_scaled.shape[1]} to {self.pca.n_components_}')

    def plot_data(self):
        pass

    def setup_train_NN_torch(self, batch_size=32, hidden_dim1=64, hidden_dim2=64, learning_rate=5e-3, lr_scheduler=True,
                             weight_decay=0.0):
        # use pytorch to train a neural network
        # split data into training and validation set
        X_train_np, X_test_np, Y_train_np, Y_test_np = sklearn.model_selection.train_test_split(self.X_preprocessed,
                                                                                                self.Y_scaled,
                                                                                                test_size=0.2,
                                                                                                random_state=1848)
        # convert data to torch tensors
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32)#.to('cuda')
        self.Y_train = torch.tensor(Y_train_np, dtype=torch.float32)#.to('cuda')

        self.X_val = torch.tensor(X_test_np, dtype=torch.float32)#.to('cuda')
        self.Y_val = torch.tensor(Y_test_np, dtype=torch.float32)#.to('cuda')

        # training parameters
        self.batch_size = batch_size
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate

        # create dataset
        train_dataset = TimeSeriesDataset(self.X_train, self.Y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.Y_val)

        # data loader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                            shuffle=True)  # , num_workers=20, persistent_workers=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                          shuffle=False)  # , num_workers=8, persistent_workers=True)

        # create model
        input_size = self.X_train.shape[1]
        output_size = self.Y_train.shape[1]
        hidden_dim = self.hidden_dim

        self.torch_model = torch_model(input_size, output_size, hidden_dim1, hidden_dim2)#.to('cuda')
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.learning_rate,
                                          weight_decay=weight_decay)

        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                                        patience=30, verbose=True)

    def train_NN_torch(self, n_epochs):
        # lists for losses
        self.train_losses = []
        self.val_losses = []
        self.param_norms = []
        self.max_norms = []
        self.min_norms = []

        # store best model weights
        best_loss = float('inf')
        self.best_model_weights = self.torch_model.state_dict()

        # training loop
        for epoch in range(n_epochs):
            self.torch_model.train()
            train_loss = 0.0
            for inputs, targets in self.train_dataloader:
                # forward pass
                outputs = self.torch_model(inputs)
                loss = self.criterion(outputs, targets)

                # backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # average train loss
            train_loss /= len(self.train_dataloader)
            self.train_losses.append(train_loss)

            # validation loss
            self.torch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_dataloader:
                    # forward pass
                    outputs = self.torch_model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            # average validation loss
            val_loss /= len(self.val_dataloader)
            self.val_losses.append(val_loss)

            # compute parameter norm
            total_norm = 0
            max_norm = 0
            min_norm = float('inf')
            for p in self.torch_model.parameters():
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                min_norm = min(min_norm, param_norm.item())
            total_norm = total_norm ** 0.5
            self.max_norms.append(max_norm)
            self.param_norms.append(total_norm)
            self.min_norms.append(min_norm)

            # save best model weights
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model_weights = self.torch_model.state_dict()

            # update learning rate
            self.scheduler.step(val_loss)

            # print losses
            print(
                f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Param Norm: {total_norm}, Max Norm: {max_norm}, Min Norm: {min_norm}, Frac: {max_norm / min_norm}')

        # load best model weights
        self.torch_model.load_state_dict(self.best_model_weights)

        # print losses of best model
        print(f'Best model, Val Loss: {best_loss}')

        # plot losses
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train_NN_keras(self):
        pass


class DataBasedModel:
    def __init__(self, casadi_model, approx_obj, cont_model):
        self.mpc_obj = None
        self.model_obj = None
        self.casadi_model = casadi_model
        self.approx_obj = approx_obj
        self.cont_model = cont_model

    def setup_model(self):
        if self.cont_model == 'MSMPR':
            self.model_obj = setup_MSMPR_model(self.casadi_model, self.approx_obj)
        elif self.cont_model == 'Tubular':
            self.model_obj = setup_tubular_model(self.casadi_model, self.approx_obj)
        else:
            print('Only MSMPR and Tubular models valid')

    def setup_controller(self, t_step=5.0):
        if self.cont_model == 'MSMPR':
            setup_controller_MSMPR(self, t_step=t_step)
        elif self.cont_model == 'Tubular':
            setup_controller_Tubular(self, t_step=t_step)
        else:
            print('Only MSMPR and Tubular models valid')

    def simulate(self):
        pass

    def control(self):
        pass


def setup_controller_MSMPR(db_model, t_step):
    db_model.mpc_obj = do_mpc.controller.MPC(db_model.model_obj)

    # setup controller
    db_model.mpc_obj.settings.n_horizon = 10
    db_model.mpc_obj.settings.t_step = t_step
    db_model.mpc_obj.settings.store_full_solution = True
    db_model.mpc_obj.settings.use_terminal_bounds = True
    db_model.mpc_obj.settings.nlpsol_opts = {'ipopt.max_iter': 2000}

    # cost function
    lterm = db_model.model_obj.aux['set_size'] + db_model.model_obj.aux['maximize_feed'] + db_model.model_obj.aux[
        'maximize_supersat']  # stage cost
    mterm = db_model.model_obj.aux['set_size']  # terminal cost

    db_model.mpc_obj.set_objective(lterm=lterm, mterm=mterm)
    cost_rterm = 3e6  # 1e6
    db_model.mpc_obj.set_rterm(F_j=cost_rterm, F_feed=2e1 * cost_rterm)

    # state constraints
    db_model.mpc_obj.set_nl_cons('supersat',
                                 expr=db_model.model_obj.aux['supersat'] - db_model.model_obj.tvp['max_supersat'],
                                 ub=0, soft_constraint=True, penalty_term_cons=1e8)

    # input constraints
    db_model.mpc_obj.bounds['lower', '_u', 'F_j'] = 1e-3
    db_model.mpc_obj.bounds['upper', '_u', 'F_j'] = 1e-1
    db_model.mpc_obj.bounds['lower', '_u', 'F_feed'] = 1e-2
    db_model.mpc_obj.bounds['upper', '_u', 'F_feed'] = 3e-2

    # time varying state constraints
    tvp_template = db_model.mpc_obj.get_tvp_template()

    def tvp_fun(t_now):
        ind = t_now
        tvp_template['_tvp', :, 'max_supersat'] = 0.03
        tvp_template['_tvp', :, 'set_L43'] = 0.0008
        # if ind <= 50:
        #     tvp_template['_tvp', : ,'size_lower'] = 0.2
        #     tvp_template['_tvp', : ,'size_upper'] = 0.35
        # elif ind <= 850:
        #     tvp_template['_tvp', : ,'size_lower'] = linear(ind-50, 0.2, 0.4)
        #     tvp_template['_tvp', : ,'size_upper'] = linear(ind-50, 0.35, 0.42)
        # else:
        #     tvp_template['_tvp', : ,'size_lower'] = 0.4
        #     tvp_template['_tvp', : ,'size_upper'] = 0.42
        return tvp_template

    db_model.mpc_obj.set_tvp_fun(tvp_fun)

    db_model.mpc_obj.setup()


def setup_controller_Tubular(db_model, t_step):
    db_model.mpc_obj = do_mpc.controller.MPC(db_model.model_obj)

    # setup controller
    db_model.mpc_obj.settings.n_horizon = 6
    db_model.mpc_obj.settings.t_step = t_step
    db_model.mpc_obj.settings.store_full_solution = True
    db_model.mpc_obj.settings.use_terminal_bounds = True
    db_model.mpc_obj.settings.nlpsol_opts = {'ipopt.max_iter': 2000}
    # db_model.mpc_obj.settings.set_linear_solver('ma27')

    # cost function
    lterm = db_model.model_obj.aux['set_size'] + db_model.model_obj.aux['maximize_feed']  # stage cost
    mterm = db_model.model_obj.aux['set_size']  # terminal cost

    db_model.mpc_obj.set_objective(lterm=lterm, mterm=mterm)
    cost_rterm = 1e6
    db_model.mpc_obj.set_rterm(F=cost_rterm * 1e1, F_J=cost_rterm)

    # scaling
    scale_x = np.array([1e-3, 1e-3, 1e2, 1e-2, 1e-1])
    for k in range(6):
        db_model.mpc_obj.scaling['_x', f'x_k-{k}'] = scale_x

    scale_u = np.array([1e-3, 1e-3, 1e2, 1e2, 1e-2, 1e2])
    for k in range(5):
        db_model.mpc_obj.scaling['_x', f'u_k-{k+1}'] = scale_u


    # state constraints
    db_model.mpc_obj.set_nl_cons('temperature',
                                 expr=-db_model.model_obj.aux['T'] + db_model.model_obj.tvp['T_constraint'],
                                 ub=0, soft_constraint=True, penalty_term_cons=1e1)

    # input constraints
    db_model.mpc_obj.bounds['lower', '_u', 'F'] = 1e-3
    db_model.mpc_obj.bounds['upper', '_u', 'F'] = 5e-3
    db_model.mpc_obj.bounds['lower', '_u', 'F_J'] = 1e-3
    db_model.mpc_obj.bounds['upper', '_u', 'F_J'] = 1e-2

    # time varying state constraints
    tvp_template = db_model.mpc_obj.get_tvp_template()

    def tvp_fun(t_now):
        ind = t_now
        tvp_template['_tvp', :, 'T_constraint'] = 310
        tvp_template['_tvp', :, 'set_size'] = 0.0015
        return tvp_template

    db_model.mpc_obj.set_tvp_fun(tvp_fun)

    db_model.mpc_obj.setup()


def setup_MSMPR_model(casadi_model, approx_obj):
    if approx_obj.last_element_only:
        # for full state prediction
        no_states = 6
        no_inputs = 2 + (approx_obj.no_discr) * 2  # 3 at inlet, 2 for each stage
    else:
        no_states = 6 * approx_obj.no_discr
        no_inputs = 2 + (approx_obj.no_discr) * 2  # 3 at inlet, 2 for each stage
    return MSMPR_model.data_based_model(casadi_model, approx_obj.no_discr, approx_obj.last_element_only, approx_obj.l,
                                        no_states, no_inputs)


def setup_tubular_model(casadi_model, approx_obj):
    no_states = 5
    no_inputs = 6
    return tubular_model.data_based_model(casadi_model, approx_obj.l, no_states, no_inputs)


def setup_simulator(model, t_step=5.0, integration_tool='cvodes', data_based=False):
    simulator_obj = do_mpc.simulator.Simulator(model)
    simulator_obj.set_param(t_step=t_step)
    simulator_obj.settings.integration_tool = integration_tool
    simulator_obj.settings.abstol = 1e-10
    simulator_obj.settings.reltol = 1e-10

    tvp_template = simulator_obj.get_tvp_template()

    def tvp_fun(t_now):
        return tvp_template

    simulator_obj.set_tvp_fun(tvp_fun)

    simulator_obj.setup()
    return simulator_obj


class Evaluate_model:
    def __init__(self, approx_obj):
        self.MSE_closed = None
        self.y_pred_closed = None
        self.MSE_open = None
        self.y_pred_open = None
        self.approx_obj = approx_obj

    def open_loop_test(self, states, inputs, plot=True):
        # test the model prediction for each time step
        y_pred_list = []
        MSE_list = []

        n_steps = states.shape[0]
        for t_i in range(self.approx_obj.l + 1, n_steps):
            # construct input for model
            x, _ = narx_data(states[t_i - self.approx_obj.l - 1:t_i, :], inputs[t_i - self.approx_obj.l - 1:t_i, :],
                             l=self.approx_obj.l)

            # scale input
            x_scaled = self.approx_obj.scaler_X.transform(x)

            # pca
            x_preprocessed = self.approx_obj.pca.transform(x_scaled)

            # convert to torch tensor
            x_torch = torch.tensor(x_preprocessed, dtype=torch.float32)

            # predict
            with torch.no_grad():
                y_pred = self.approx_obj.torch_model(x_torch)

            # inverse transform
            y_pred = self.approx_obj.scaler_Y.inverse_transform(y_pred.numpy())

            y_pred_list.append(y_pred)
            MSE_list.append(np.mean((states[t_i, :] - y_pred) ** 2))  # t_i or t_i+1?

            self.y_pred_open = np.array(y_pred_list).squeeze()
            self.MSE_open = np.array(MSE_list)

        if plot:
            fig, ax = plt.subplots(self.approx_obj.n_states, 1, figsize=(10, 40))

            for i in range(self.approx_obj.n_states):
                ax[i].plot(states[self.approx_obj.l + 1:, i], label='True')
                ax[i].plot(self.y_pred[:, i], label='Predicted')
                ax[i].legend()

            plt.show()

    def closed_loop_test(self, states, inputs, plot=True, casadi_model=None, test_mpc_model=None, save_data=False):
        # test the model prediction for each time step and use the prediction as input for the next time step
        # model predicts the next state based on the last l states and inputs
        # prediction of model with input from data is used as input for the next prediction

        # predictions are made with list of states, which is initialized with the first l states and updated with each prediction
        # inputs are taken from the data

        # initialize list of states
        states_array = states[:self.approx_obj.l, :]

        # list of predictions
        y_pred_list = []

        # list of MSE
        MSE_list = []

        # number of time steps
        n_steps = states.shape[0]

        for t_i in range(self.approx_obj.l, n_steps):
            # construct input for current step
            x = narx_data_input(states_array[-self.approx_obj.l:, :], inputs[t_i - self.approx_obj.l:t_i, :],
                                l=self.approx_obj.l)

            # # scale input
            # x_scaled = self.approx_obj.scaler_X.transform(x)
            #
            # # pca
            # x_preprocessed = self.approx_obj.pca.transform(x_scaled)
            #
            # # convert to torch tensor
            # x_torch = torch.tensor(x_preprocessed, dtype=torch.float32)
            #
            # # predict
            # with torch.no_grad():
            #     y_pred = self.approx_obj.torch_model(x_torch)
            #
            # # inverse transform
            # y_pred = self.approx_obj.scaler_Y.inverse_transform(y_pred.numpy())

            ###########################
            y_pred = casadi_model(x)

            # # take states from x
            # states_test = x[:, :self.approx_obj.l*10]
            # inputs_test = x[:, self.approx_obj.l*10:]
            # inputs_current = inputs_test[:,:7]
            # inputs_past = inputs_test[:,7:]
            #
            # states_in = np.concatenate([states_test, inputs_past], axis=1)
            # y_pred = test_mpc_model._rhs_fun(states_in,inputs_current,test_mpc_model._z(0),test_mpc_model._tvp(0),test_mpc_model._p(0),test_mpc_model._w(0)).T
            ############################

            # append prediction to list
            y_pred_list.append(y_pred)

            # append MSE to list
            MSE_list.append(np.mean((states[t_i, :] - y_pred) ** 2))

            # update list of states
            states_array = np.append(states_array, y_pred.T, axis=0)

        # convert list to numpy array
        self.y_pred_closed = np.array(y_pred_list).squeeze()
        self.MSE_closed = np.array(MSE_list)

        # print MSE
        print(f'MSE: {np.mean(self.MSE_closed)}')

        if plot:
            fig, ax = plt.subplots(states.shape[1], 1, figsize=(5, 9))

            for i in range(states.shape[1]):
                ax[i].plot(states[self.approx_obj.l:, i], label='True')
                ax[i].plot(self.y_pred_closed[:, i], label='Predicted')
                ax[i].legend()

            plt.show()

        if save_data:
            data = {
                'states': states,
                'inputs': inputs,
                'y_pred_closed': self.y_pred_closed,
                'MSE_closed': self.MSE_closed
            }
            with open('data_closed_loop_testing.pkl', 'wb') as f:
                pickle.dump(data, f)


def narx_data(states, inputs, l):
    # go through lists of states and inputs and construct data matrix
    X_full = []
    Y_full = []
    for list_i in range(len(states)):
        data_matrix = np.hstack((states[list_i], inputs[list_i]))

        # number of time steps
        n_time_steps = data_matrix.shape[0]

        # number of states
        n_states = states[list_i].shape[1]

        # construct X and Y
        X = []
        Y = []

        for t in range(n_time_steps - l):
            X_i = []
            U_i = []
            for i in reversed(range(l)):
                X_i.append(states[list_i][i + t, :].reshape(1, -1))
                U_i.append(inputs[list_i][i + t, :].reshape(1, -1))
            X.append(np.hstack((*X_i, *U_i)))
            Y.append(states[list_i][t + l, :n_states])

        X = np.vstack(X)
        Y = np.vstack(Y)

        X_full.append(X)
        Y_full.append(Y)

    X_full = np.vstack(X_full)
    Y_full = np.vstack(Y_full)

    # data matrix containing all states and inputs
    # data_matrix = np.concatenate([states, inputs], axis=1)
    # data_matrix = ca.horzcat(states, inputs)
    #
    # # number of time steps
    # n_time_steps = data_matrix.shape[0]
    #
    # # number of states
    # n_states = states.shape[1]
    #
    # # construct X and Y
    # X = []
    # Y = []
    #
    # for t in range(n_time_steps - l):
    #     X_i = []
    #     U_i = []
    #     for i in reversed(range(l)):
    #         X_i.append(states[i+t, :].reshape(1,-1))
    #         U_i.append(inputs[i+t, :].reshape(1,-1))
    #     X.append(ca.horzcat(*X_i, *U_i))
    #     Y.append(states[t + l, :n_states])
    # X = ca.vertcat(*X)
    # Y = ca.vertcat(*Y)

    # for t in range(l - 1, n_time_steps - 1):
    #     input_vector = []
    #     for i in range(l):
    #         input_vector.append(data_matrix[t - i, :])
    #     X.append(np.concatenate(input_vector).reshape(1, -1))
    #     Y.append(data_matrix[t + 1, :n_states].reshape(1, -1))
    #
    # X = np.concatenate(X, axis=0)
    # Y = np.concatenate(Y, axis=0)

    return X_full, Y_full


def narx_data_input(states, inputs, l):
    data_matrix = np.hstack((states, inputs))

    # number of time steps
    n_time_steps = data_matrix.shape[0]

    # number of states
    n_states = states.shape[1]

    # construct X and Y
    X = []
    Y = []

    for t in range(n_time_steps - l + 1):
        X_i = []
        U_i = []
        for i in reversed(range(l)):
            X_i.append(states[i + t, :].reshape(1, -1))
            U_i.append(inputs[i + t, :].reshape(1, -1))
        X.append(np.hstack((*X_i, *U_i)))

    X = np.vstack(X)

    # # data matrix containing all states and inputs
    # data_matrix = np.concatenate([states, inputs], axis=1)
    #
    # # number of time steps
    # n_time_steps = data_matrix.shape[0]
    #
    # # number of states
    # n_states = states.shape[1]
    #
    # # construct X
    # X = []
    #
    # for t in range(l - 1, n_time_steps):
    #     input_vector = []
    #     for i in range(l):
    #         input_vector.append(data_matrix[t - i, :])
    #     X.append(np.concatenate(input_vector).reshape(1, -1))
    #
    # X = np.concatenate(X, axis=0)
    return X


def scale_data(X, Y):
    # use sklearn data scaler
    scaler_X = sklearn.preprocessing.StandardScaler()
    scaler_Y = sklearn.preprocessing.StandardScaler()

    # scaler_X = sklearn.preprocessing.MinMaxScaler()
    # scaler_Y = sklearn.preprocessing.MinMaxScaler()

    # fit scalers
    scaler_X.fit(X)
    scaler_Y.fit(Y)

    # for standard scaler
    # check if variance is 0 for both scalers and set to 1 if it is
    scaler_X.var_[scaler_X.var_ < 1e-20] = 1
    scaler_Y.var_[scaler_Y.var_ < 1e-20] = 1

    # for standard scaler
    X_scaled = (X - scaler_X.mean_) / np.sqrt(scaler_X.var_)
    Y_scaled = (Y - scaler_Y.mean_) / np.sqrt(scaler_Y.var_)

    # # for minmax scaler
    # # check if variance is 0 for both scalers and set to 1 if it is
    # scaler_X.data_range_[scaler_X.data_range_ < 1e-20] = 1
    # scaler_X.data_range_[scaler_X.data_range_ < 1e-20] = 1
    #
    # # transform data
    #
    # # for minmax scaler
    # X_scaled = (X - scaler_X.data_min_) / scaler_X.data_range_
    # Y_scaled = (Y - scaler_Y.data_min_) / scaler_Y.data_range_

    return scaler_X, scaler_Y, X_scaled, Y_scaled


def PCA_data(X_scaled, n_components=10):
    # use sklearn PCA
    pca = sklearn.decomposition.PCA(n_components=n_components)

    # fit and transform data
    X_preprocessed = pca.fit_transform(X_scaled)

    # print explained variance
    print(
        f'{n_components} principal components containing {np.sum(pca.explained_variance_ratio_)} of variance')

    return pca, X_preprocessed


def torch_to_casadi_function(model: torch.nn.Module, approx_obj, activation: str = "sigmoid"):
    """
    Converts a PyTorch model to a CasADi function for evaluation.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        approx_obj: Approximate object containing data for scaling and PCA.
        activation (str): The activation function to use ("sigmoid" or "relu"). Default is "sigmoid".

    Returns:
        casadi.Function: A CasADi function that evaluates the PyTorch model.
    """

    # Automatically extract input size from the first Linear layer
    layers = list(model.children())

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    if len(linear_layers) == 0:
        raise ValueError("The model does not contain any Linear layers. Cannot create CasADi function.")

    # Input size is taken from the first Linear layer
    input_size = approx_obj.X.shape[1]

    # CasADi symbolic input
    x = ca.SX.sym('x', input_size)

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input and pca
    # for standard scaler
    input_casadi = (input_casadi - approx_obj.scaler_X.mean_) / np.sqrt(approx_obj.scaler_X.var_)

    # # for minmax scaler
    # input_casadi = (input_casadi - approx_obj.scaler_X.data_min_) / approx_obj.scaler_X.data_range_

    input_casadi = ca.reshape(ca.mtimes(ca.reshape(input_casadi, 1, -1), approx_obj.pca.components_.T), -1, 1)

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract the weights and bias from the PyTorch linear layer
        weight = layer.weight.detach().numpy()#.detach().cpu().numpy()
        bias = layer.bias.detach().numpy()#.detach().cpu().numpy()

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight, input_casadi) + bias

        # Apply activation function unless it's the last Linear layer
        if i < len(linear_layers) - 1:
            if activation == "relu":
                input_casadi = ca.fmax(input_casadi, 0)  # ReLU activation
            elif activation == "sigmoid":
                input_casadi = 1 / (1 + ca.exp(-input_casadi))  # Sigmoid activation
            elif activation == "tanh":
                input_casadi = ca.tanh(input_casadi)
            else:
                raise ValueError("Unsupported activation function. Use 'sigmoid', 'relu', or 'tanh'.")

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * np.sqrt(approx_obj.scaler_Y.var_) + approx_obj.scaler_Y.mean_

    # # for minmax scaler
    # input_casadi = input_casadi * approx_obj.scaler_Y.data_range_ + approx_obj.scaler_Y.data_min_

    # Create the CasADi function
    model_casadi_function = ca.Function('model', [x], [input_casadi])

    return model_casadi_function


def mean_width_moments(PBE_state):
    # get number of stages or discrete elements of tubular
    n_discrete = int(PBE_state.shape[1] / 6)

    mean = np.zeros((PBE_state.shape[0], n_discrete))
    width = np.zeros((PBE_state.shape[0], n_discrete))

    # get L43 and L43*CV for each stage
    for elem_i in range(n_discrete):
        mean[:, elem_i] = PBE_state[:, 4 + 6 * elem_i] / PBE_state[:, 3 + 6 * elem_i]
        width[:, elem_i] = np.sqrt(PBE_state[:, 5 + 6 * elem_i] / PBE_state[:, 3 + 6 * elem_i] - (
                PBE_state[:, 4 + 6 * elem_i] / PBE_state[:, 3 + 6 * elem_i]) ** 2)

    return mean, width


def mean_width_DPBE(PBE_state, no_disc, no_disc_PBE, L_i):
    # calculate median and width of DPBE
    # median: d50
    # width: d90 - d10

    # reshape PBE_state to get all DPBE values for each stage
    L_i = np.array(L_i)
    t_steps = PBE_state.shape[0]
    PBE_state = PBE_state.reshape(t_steps, no_disc, no_disc_PBE)

    mean = np.zeros((PBE_state.shape[0], no_disc))
    width = np.zeros((PBE_state.shape[0], no_disc))

    def find_x_at_percentile(x, y, percentile):
        y_normalized = np.cumsum(y) / np.sum(y)
        idx_below = np.max(np.where(y_normalized < percentile)[0])
        idx_above = idx_below + 1

        x0, x1 = x[idx_below], x[idx_above]
        y0, y1 = y_normalized[idx_below], y_normalized[idx_above]

        return x0 + (percentile - y0) * (x1 - x0) / (y1 - y0)

    for t_i in range(t_steps):
        for elem_i in range(no_disc):
            # find d50
            d50 = find_x_at_percentile(L_i, PBE_state[t_i, elem_i, :], 0.5)
            mean[t_i, elem_i] = d50

            # find d10 and d90
            d10 = find_x_at_percentile(L_i, PBE_state[t_i, elem_i, :], 0.1)
            d90 = find_x_at_percentile(L_i, PBE_state[t_i, elem_i, :], 0.9)

            width[t_i, elem_i] = d90 - d10

    return mean, width


def setup_physical_model(model_type):
    if model_type == 'MSMPR':
        model_obj = setup_MSMPR()
    elif model_type == 'Tubular':
        model_obj = setup_tubular()

    return model_obj


def setup_MSMPR():
    pass


def setup_tubular():
    pass


def MPC_input(x, model_type, method_type, approx_obj, model_obj):
    # calculate input for data based model based on output of physical model
    # necessary because: pyhsical model has different states than data based model, NARX
    if model_type == 'MSMPR':
        x_MPC = MPC_input_MSMPR(x, method_type, approx_obj)
    elif model_type == 'Tubular':
        x_MPC = MPC_input_tubular(x, method_type, approx_obj, model_obj)
    return x_MPC


def MPC_input_MSMPR(x, method_type, approx_obj):
    no_stages = int(x.shape[0] / 9)

    x_PBE = np.array(x.T[:6 * no_stages])

    x_cont = np.array(x.T[6 * no_stages:])  # full continuous state

    # get single continuous states for cases where only the last element is used
    T = np.array(x.T[6 * no_stages:7 * no_stages])
    T_j = np.array(x.T[7 * no_stages:8 * no_stages])
    c = np.array(x.T[8 * no_stages:9 * no_stages])

    if method_type == 'SMOM' or method_type == 'QMOM':
        mean, width = mean_width_moments(x_PBE)
    elif method_type == 'DPBE':
        print('Convert DPBE to mean and width not implemented yet')
    elif method_type == 'OCFE':
        print('Convert OCFE to mean and width not implemented yet')

    sol = cryst.solubility(T)
    rel_S = c / sol - 1

    if approx_obj.last_element_only:
        mean = mean[:, -1].reshape(1, -1)
        width = width[:, -1].reshape(1, -1)
        x_cont = np.array([T[:, -1], T_j[:, -1], c[:, -1]]).reshape(1, -1)

    # return np.concatenate([mean, width, x_cont, np.max(rel_S).reshape(-1, 1)], axis=1)
    return np.concatenate([mean, x_cont, np.array(rel_S)], axis=1)  # only mean and all supersat


def MPC_input_tubular(x, method_type, approx_obj, model_obj):
    no_states = 3
    if method_type == 'SMOM' or method_type == 'QMOM':
        print('SMOM and QMOM not implemented for Tubular model')
    elif method_type == 'DPBE':
        n_PBE = model_obj.PBE_param['no_class']
        L_i = model_obj.PBE_obj.L_i
        denominator = n_PBE + no_states
    elif method_type == 'OCFE':
        print('OCFE not implemented for Tubular model')
    no_stages = int(x.shape[0] / denominator)

    x_PBE = np.array(x[:-(no_stages * no_states)])
    x_cont = np.array(x[-(no_stages * no_states):])  # full continuous state

    # get single continuous states for cases where only the last element is used
    T = x_cont[:no_stages].T
    T_j = x_cont[no_stages:2 * no_stages].T
    c = x_cont[2 * no_stages:3 * no_stages].T

    if method_type == 'SMOM' or method_type == 'QMOM':
        mean, width = mean_width_moments(x_PBE)
    elif method_type == 'DPBE':
        mean, width = mean_width_DPBE(x_PBE.T, no_stages, n_PBE, L_i)
    elif method_type == 'OCFE':
        print('Convert OCFE to mean and width not implemented yet')

    if approx_obj.last_element_only:
        mean = mean[:, -1].reshape(1, -1)
        width = width[:, -1].reshape(1, -1)
        x_cont = np.array([T[:, -1], T_j[:, -1], c[:, -1]]).reshape(1, -1)

    return np.concatenate([mean, width, x_cont], axis=1)
    # return np.concatenate([mean, x_cont], axis=1)


def full_input_for_simulator(u0, model_obj, method_i, approx_obj):
    # adds values to inputs from data based model to get full input for simulator
    model_type = model_obj.model_type
    if model_type == 'MSMPR':
        no_stages = model_obj.method_param['no_stages']
        # current: inputs for db: F_j x no_stages, F_feed
        F_j = u0[:no_stages].reshape(-1, 1)
        F_feed = u0[no_stages].reshape(-1, 1)

        # needed for simulator: T_j_in x no_stages, F_j x no_stages, F_feed, T_feed
        T_j_in = 295 * np.ones((no_stages, 1))
        T_feed = np.array([[323.15]])
        return np.concatenate((T_j_in, F_j, F_feed, T_feed))
    elif model_type == 'Tubular':
        c_in = np.array([[0]])
        T_in = np.array([[350]])
        T_j_in = np.array([[350]])
        T_env = np.array([[295]])

        F = u0[0].reshape(-1, 1)
        F_J = u0[1].reshape(-1, 1)

        return np.concatenate((F, c_in, T_in, T_j_in, F_J, T_env))


def plot_results(results):
    sim_results = results['simulator']
    # plot: _aux: L43, and CV, _x: T, T_j, c, _u: T_j_in, F_j, F_feed, c_feed, T_feed
    fig, ax = plt.subplots(10, 1, figsize=(10, 40), sharex=True)

    ax[0].plot(sim_results['_aux', 'L43'])
    ax[0].set_ylabel('L43')

    ax[1].plot(sim_results['_aux', 'CV'])
    ax[1].set_ylabel('CV')

    ax[2].plot(sim_results['_x', 'T'])
    ax[2].set_ylabel('T')

    ax[3].plot(sim_results['_x', 'T_j'])
    ax[3].set_ylabel('T_j')

    ax[4].plot(sim_results['_x', 'c'])
    ax[4].set_ylabel('c')

    ax[5].plot(sim_results['_u', 'T_j_in'])
    ax[5].set_ylabel('T_j_in')

    ax[6].plot(sim_results['_u', 'F_j'])
    ax[6].set_ylabel('F_j')

    ax[7].plot(sim_results['_u', 'F_feed'])
    ax[7].set_ylabel('F_feed')

    ax[8].plot(sim_results['_u', 'c_feed'])
    ax[8].set_ylabel('c_feed')

    ax[9].plot(sim_results['_u', 'T_feed'])
    ax[9].set_ylabel('T_feed')

    fig.tight_layout()

    plt.show()


def initialize_narx(x0, u0, approx_obj):
    # return vector containing l times the initial state and initial input
    if approx_obj.l > 1:
        x_full = np.concatenate(np.array([x0.T for _ in range(approx_obj.l)]))
        u_full = np.concatenate(np.array([u0.cat for _ in range(approx_obj.l - 1)]))
        return np.vstack((x_full, u_full))
    else:
        return x0


def narx(x_MPC, x_MPC_0, u, l):
    n_u = u.shape[0]
    n_x = x_MPC_0.shape[1]

    x_old = x_MPC[:n_x * l]
    u_old = x_MPC[n_x * l:]

    if l > 1:
        x_new = np.concatenate((x_MPC_0.T, x_old[:n_x * (l - 1)], u, u_old[:n_u * (l - 2)]))
    else:
        x_new = x_MPC_0.T
    return x_new


def train_model(name, l, n_components, learning_rate, hidden_dim1, hidden_dim2, batch_size, n_epochs, save_model=False,
                last_element_only=False, weight_decay=0.0):
    # list of simulation objects and meta data from name_list
    simulation_obj_list = []
    meta_data_sim_list = []

    for name_i in name:
        simulation_obj = do_mpc.data.load_results('./results/' + name_i + '.pkl')
        # load meta_data from json
        with open('./results/' + name_i + '_meta_data.json', 'r') as f:
            meta_data_sim = json.load(f)
        simulation_obj_list.append(simulation_obj)
        meta_data_sim_list.append(meta_data_sim)

    # create approximate object
    approx_obj = Approximate()

    # load data
    approx_obj.load_data(simulation_obj_list, meta_data_sim_list, last_element_only=last_element_only)

    # setup data
    approx_obj.setup_data(l=l, n_components=n_components)

    # train neural network
    approx_obj.setup_train_NN_torch(learning_rate=learning_rate, hidden_dim1=int(hidden_dim1),
                                    hidden_dim2=int(hidden_dim2), batch_size=batch_size,
                                    weight_decay=weight_decay)
    approx_obj.train_NN_torch(n_epochs=n_epochs)

    # transform torch model to casadi
    casadi_model = torch_to_casadi_function(approx_obj.torch_model, approx_obj, activation='sigmoid')

    # save casadi model and approximate object
    if save_model:
        with open('casadi_model.pkl', 'wb') as f:
            pickle.dump(casadi_model, f)
            pickle.dump(approx_obj, f)
            pickle.dump(meta_data_sim, f)

    return casadi_model, approx_obj, [meta_data_sim]


if __name__ == "__main__":
    # model
    cont_model = 'MSMPR'
    PBE_method = 'SMOM'

    # either train a new model or load an existing model
    load_model = True

    # save model
    save_model = True

    # last element only
    last_element_only = True

    # folder
    if cont_model == 'MSMPR':
        folder = 'SMOM/Time step 10/'
    elif cont_model == 'Tubular':
        folder = 'DPBE/Timestep 10 better sampling reduced inputs/'

    # load simulation object
    if cont_model == 'MSMPR':
        name_list = [folder + 'MSMPR_SMOM_train']
    elif cont_model == 'Tubular':
        name_list = [folder + 'Tubular_DPBE_Train1', folder + 'Tubular_DPBE_Train2', folder + 'Tubular_DPBE_Train3',
                     folder + 'Tubular_DPBE_Train4', folder + 'Tubular_DPBE_Train5', folder + 'Tubular_DPBE_Train6',
                     folder + 'Tubular_DPBE_Train7', folder + 'Tubular_DPBE_Train8', folder + 'Tubular_DPBE_Train9']

    if load_model:
        # load approximate object
        if cont_model == 'MSMPR':
            with open('run1casadi_model.pkl', 'rb') as f:
                casadi_model = pickle.load(f)
                approx_obj = pickle.load(f)
                meta_data_sim = [pickle.load(f)]
        elif cont_model == 'Tubular':
            with open('run1casadi_model.pkl', 'rb') as f:
                casadi_model = pickle.load(f)
                approx_obj = pickle.load(f)
                meta_data_sim = [pickle.load(f)]

    else:
        if cont_model == 'MSMPR':
            l = 10  # tubular 40 # 25
            n_components = 0.999999999999
            learning_rate = 1e-2
            hidden_dim1 = 250
            hidden_dim2 = 250
            batch_size = 256
            n_epochs = 1000
            weight_decay = 5e-8
        elif cont_model == 'Tubular':
            l = 6
            n_components = 0.999999999999
            learning_rate = 1e-2
            hidden_dim1 = 250
            hidden_dim2 = 250
            batch_size = 256
            n_epochs = 1
            weight_decay = 8e-7

        casadi_model, approx_obj, meta_data_sim = train_model(name_list, l, n_components, learning_rate, hidden_dim1,
                                                              hidden_dim2,
                                                              batch_size, n_epochs,
                                                              save_model=save_model,
                                                              last_element_only=last_element_only,
                                                              weight_decay=weight_decay)

    # test model
    # load data and get states and inputs
    approx_test_obj = Approximate()
    if cont_model == 'MSMPR':
        name_test = folder + 'MSMPR_SMOM_test'
    elif cont_model == 'Tubular':
        name_test = folder + 'Tubular_DPBE_Test'

    test_simulation_obj = [do_mpc.data.load_results('./results/' + name_test + '.pkl')]
    approx_test_obj.load_data(test_simulation_obj, meta_data_sim, last_element_only=last_element_only)
    test_states = approx_test_obj.states[0][500:1000, :]
    test_inputs = approx_test_obj.inputs[0][500:1000, :]

    # evaluate model
    eval_obj = Evaluate_model(approx_obj)
    # eval_obj.open_loop_test(test_states, test_inputs)
    eval_obj.closed_loop_test(test_states, test_inputs, casadi_model=casadi_model, save_data=False)
    run_mpc = True
    if run_mpc:
        # x = np.arange(10).reshape(-1,1)*0.1
        # u = np.arange(10).reshape(-1,1)*10
        # narx_data(x,u,3)

        # use_mpc_model = 'data_based'
        use_mpc_model = 'physical'

        # create data based model object
        data_based_model = DataBasedModel(casadi_model, approx_obj, cont_model=cont_model)

        # setup do-mpc model
        data_based_model.setup_model()

        # create physical model object
        model_i = cont_model
        method_i = PBE_method

        model_obj = model_class.Model(model_i)
        print(model_obj)

        # load distribution parameters from python file in config folder
        dist_params = config_data.distribution_params(model_i)

        # load PBE parameters from python file in config folder
        PBE_param = config_data.load(method_i, **dist_params)
        n_initial = config_data.load('n_initial', **dist_params)

        # setup PBE
        model_obj.setup_PBE(PBE_param, **n_initial, **{'G': 0.0, 'beta': 0.0})

        # load cryst parameters from python file in config folder
        cryst_param = config_data.load('cryst_param', **dist_params)

        # load parameters for continuous model (MSMPR, Tubular) from python file in config folder
        cont_param = config_data.load(model_i, **dist_params)

        # setup model
        model_obj.setup_model(**cryst_param, **cont_param)

        # setup simulator
        simulator_physical_obj = setup_simulator(model_obj.model_obj, t_step=1.0 + 0 * meta_data_sim[0]['t_step'],
                                                 data_based=False)

        # setup controller
        # data based
        if use_mpc_model == 'data_based':
            data_based_model.setup_controller(t_step=meta_data_sim[0]['t_step'])
        # physical
        if use_mpc_model == 'physical':
            model_obj.setup_controller(t_step=meta_data_sim[0]['t_step'])

        # set initial guess
        if use_mpc_model == 'data_based':
            data_based_model.mpc_obj.set_initial_guess()
        if use_mpc_model == 'physical':
            model_obj.mpc_obj.set_initial_guess()

        # initial state for simulator
        x0 = simulator_physical_obj.x0
        u0 = simulator_physical_obj.u0
        initial_state = config_data.initial_cont(model_i)

        model_class.set_initial_state(x0, u0, model_obj.model_obj, steady_state=True,
                                      **{'model_type': model_obj.model_type, 'PBE_initial': model_obj.PBE_initial,
                                         'method': model_obj.PBE_obj.method}, **model_obj.method_param, **initial_state)

        # set optimizer values for physical model MPC (necessary e.g. for QMOM to avoid division by zero)
        if use_mpc_model == 'physical':
            u0_try = np.array([295,295, 0.001, 0.001, 0.02, 323.15])
            model_class.setup_opt_x_num(model_obj.mpc_obj, x0, u0_try)
            model_class.setup_opt_aux_num(model_obj.mpc_obj)

        # run simulation with data based model for controller
        t_steps = 100

        x_MPC_0 = MPC_input(x0.cat, model_i, method_i, approx_obj, model_obj)
        x_MPC = initialize_narx(x_MPC_0, u0, approx_obj)

        # x_previous = x_MPC.T[:-10]
        # x0_data_based['previous_x'] = x_previous

        u = np.array(u0.cat)

        # eval_obj.closed_loop_test(test_states, test_inputs, casadi_model=casadi_model,
        #                           test_mpc_model=simulator_data_based_obj.model)

        # generate random input sequence
        # sequence = model_class.generate_input_sequence(counter_distribution='uniform', sequence_length=t_steps,
        #                                                **cont_param)
        if use_mpc_model == 'physical':
            x_next = x0.cat

        mpc_time = []
        full_time = []
        for t_i in range(t_steps):
            print('Step:', t_i + 1, 'of', t_steps)

            if use_mpc_model == 'physical':
                time_before_mpc = time.perf_counter()
                u0 = model_obj.mpc_obj.make_step(x_next)
                time_after_mpc = time.perf_counter()
                mpc_time.append(time_after_mpc - time_before_mpc)
                for _ in range(10):
                    simulator_physical_obj.make_step(u0)
                x_next = simulator_physical_obj.x0.cat

                time_after_iteration = time.perf_counter()

                full_time.append(time_after_iteration - time_before_mpc)

            if use_mpc_model == 'data_based':
                # MPC
                time_before_mpc = time.perf_counter()
                u0 = data_based_model.mpc_obj.make_step(x_MPC)
                time_after_mpc = time.perf_counter()
                mpc_time.append(time_after_mpc - time_before_mpc)

                # code this in function:
                u_full = full_input_for_simulator(u0, model_obj, method_i, approx_obj)

                if model_i == 'MSMPR':
                    for _ in range(10):
                        simulator_physical_obj.make_step(u_full)

                elif model_i == 'Tubular':
                    for _ in range(10):
                        print('Simulating time step:', _)
                        simulator_physical_obj.make_step(u_full)

                x_next = simulator_physical_obj.x0.cat
                x_MPC_0 = MPC_input(x_next, model_i, method_i, approx_obj, model_obj)
                x_MPC = narx(x_MPC, x_MPC_0, u_full, approx_obj.l)

                time_after_iteration = time.perf_counter()

                full_time.append(time_after_iteration - time_before_mpc)

        #

        # #
        # plot results
        if model_i == 'MSMPR':
            time = simulator_physical_obj.data['_time']
            c = simulator_physical_obj.data['_x', 'c']
            T = simulator_physical_obj.data['_x', 'T']
            T_j = simulator_physical_obj.data['_x', 'T_j']
            L43 = simulator_physical_obj.data['_aux', 'size']
            # CV = simulator_physical_obj.data['_aux', 'width']
            rel_S = simulator_physical_obj.data['_aux', 'rel_S']
            F_j = simulator_physical_obj.data['_u', 'F_j']
            F_feed = simulator_physical_obj.data['_u', 'F_feed']

            fig, ax = plt.subplots(8, 1, figsize=(8, 12), sharex=True)

            ax[0].plot(time, c)
            ax[0].set_ylabel('c')

            ax[1].plot(time, T)
            ax[1].set_ylabel('T')

            ax[2].plot(time, T_j)
            ax[2].set_ylabel('T_j')

            ax[3].plot(time, L43)
            ax[3].hlines(8e-4, 0, time[-1], color='r', linestyle='--')
            ax[3].set_ylabel('L43')

            # ax[4].plot(time, CV)
            # ax[4].set_ylabel('CV')

            ax[5].plot(time, rel_S)
            ax[5].set_ylabel('rel_S')

            ax[6].step(time, F_j)
            ax[6].set_ylabel('F_j')

            ax[7].step(time, F_feed)
            ax[7].set_ylabel('F_feed')

            fig.legend()
            fig.tight_layout()
            fig.show()

        elif model_i == 'Tubular':
            time = simulator_physical_obj.data['_time']
            c = simulator_physical_obj.data['_x', 'c']
            T = simulator_physical_obj.data['_x', 'T']
            T_j = simulator_physical_obj.data['_x', 'T_j']
            PBE_state = simulator_physical_obj.data['_x', 'PBE_state']
            mu = simulator_physical_obj.data['_aux', 'mu']
            F = simulator_physical_obj.data['_u', 'F']
            T_j_in = simulator_physical_obj.data['_u', 'T_j_in']
            F_J = simulator_physical_obj.data['_u', 'F_J']

            size, width = mean_width_DPBE(PBE_state, model_obj.method_param['n_discr'], model_obj.PBE_param['no_class'],
                                          model_obj.PBE_obj.L_i)

            fig, ax = plt.subplots(8, 1, figsize=(8, 12), sharex=True)

            ax[0].plot(time, c)
            ax[0].set_ylabel('c')

            ax[1].plot(time, T)
            ax[1].set_ylabel('T')

            ax[2].plot(time, T_j)
            ax[2].set_ylabel('T_j')

            ax[3].plot(time, size[:, -1])
            ax[3].set_ylabel('size at outlet')

            # ax[4].plot(time, width[:, -1])
            # ax[4].set_ylabel('width at outlet')

            ax[5].plot(time, F)
            ax[5].set_ylabel('F')

            ax[6].plot(time, T_j_in)
            ax[6].set_ylabel('T_j_in')

            ax[7].plot(time, F_J)
            ax[7].set_ylabel('F_J')

            fig.legend()
            fig.tight_layout()
            fig.show()

    if use_mpc_model == 'data_based':
        # mpc meta data
        meta_data_mpc = {
            't_step_mpc': data_based_model.mpc_obj.settings.t_step,
            'n_horizon': data_based_model.mpc_obj.settings.n_horizon,
            'n_robust': data_based_model.mpc_obj.settings.n_robust,
            't_step_sim': simulator_physical_obj.settings.t_step,
            'mpc_time': mpc_time,
            'full_iter_time': full_time
        }

        if model_i == 'MSMPR':
            meta_data_db_model = {
                'l': approx_obj.l,
                'n_states': approx_obj.pca.components_.shape[1],
                'pca_components': approx_obj.pca.n_components_,
                'pca_variance': np.sum(approx_obj.pca.explained_variance_ratio_[:approx_obj.pca.components_.shape[1]]),
                'scaler_X': str(approx_obj.scaler_X),
                'scaler_Y': str(approx_obj.scaler_Y),
            }
        elif model_i == 'Tubular':
            meta_data_db_model = {
                'l': approx_obj.l,
                'n_states': approx_obj.pca.components_.shape[1],
                'pca_components': approx_obj.pca.n_components_,
                'pca_variance': np.sum(approx_obj.pca.explained_variance_ratio_[:approx_obj.pca.components_.shape[1]]),
                'scaler_X': str(approx_obj.scaler_X),
                'scaler_Y': str(approx_obj.scaler_Y),
                'L_i': model_obj.PBE_obj.L_i,
            }

        meta_data_model_training = {
            'criterion': str(approx_obj.criterion),
            'optimizer': str(approx_obj.optimizer),
            'batch_size': approx_obj.batch_size,
        }

        # save results
        do_mpc.data.save_results([data_based_model.mpc_obj, simulator_physical_obj], result_name='do_mpc')

        # save all results and objects
        results_dict = {'meta_data_train_sim': meta_data_sim[0],
                        'meta_data_mpc': meta_data_mpc,
                        'meta_data_db_model': meta_data_db_model,
                        'meta_data_model_training': meta_data_model_training,
                        }
        # pickle
        with open('./results/MPC Tubular/meta_data.pkl', 'wb') as f:
            pickle.dump(results_dict, f)


    elif use_mpc_model == 'physical':
        # mpc meta data
        meta_data_mpc = {
            't_step_mpc': model_obj.mpc_obj.settings.t_step,
            'n_horizon': model_obj.mpc_obj.settings.n_horizon,
            'n_robust': model_obj.mpc_obj.settings.n_robust,
            't_step_sim': simulator_physical_obj.settings.t_step,
            'mpc_time': mpc_time,
            'full_iter_time': full_time
        }

        do_mpc.data.save_results([model_obj.mpc_obj, simulator_physical_obj], result_name='do_mpc')

        # save all results and objects
        results_dict = {
            'meta_data_mpc': meta_data_mpc,
        }

        # pickle
        with open('./results/MPC MSMPR/meta_data.pkl', 'wb') as f:
            pickle.dump(results_dict, f)


    sys.exit()
