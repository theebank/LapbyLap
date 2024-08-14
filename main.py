import pandas as pd
import numpy as np
import torch, torch.nn as nn
import streamlit as st
import json
from urllib.request import urlopen

def get_driver_info(number, session_key='latest'):
    res = urlopen(f'https://api.openf1.org/v1/drivers?driver_number={number}&&session_key={session_key}')
    _driver = json.loads(res.read().decode('utf-8'))
    print(_driver)


def get_circuit_info(country, year=2023):
    res = urlopen(f'https://api.openf1.org/v1/meetings?year={year}&country_name={country}')
    _circuit = json.loads(res.read().decode('utf-8'))
    print(_circuit)

def get_session_info(country, year=2023):
    res = urlopen(f'https://api.openf1.org/v1/sessions?country_name={country}&year={year}')
    _sessions = json.loads(res.read().decode('utf-8'))

    print(_sessions)

def get_position_info(session_key='latest'):
    res = urlopen(f'https://api.openf1.org/v1/position?session_key={session_key}')
    _positions = json.loads(res.read().decode('utf-8'))
    print(_positions)

class RacePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, lstm_hidden, lstm_layers, dropout):
        super(RacePredictionModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden, num_layers=lstm_layers,dropout=dropout, batch_first=True)


        self.fc = nn.Linear(lstm_hidden, self.output_size)

    def zero_states(self, batch_size = 1):
        hidden_states = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden)
        cell_states = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden)
        return (hidden_states, cell_states)

    def forward(self, ins, prev_states=None):
        lstm_outs,next_states = self.lstm(ins, prev_states)
        outs = self.fc(lstm_outs)
        return outs, next_states

def time_to_int(t):
    # converts a time of the format mm:ss:ms to seconds
    if(t==float):
        return t
    t2= str(t)
    ts = t2.rsplit(":")
    if('\\N' in t2):
        return None
    if(not '.' in t2):
        return None
    if(len(ts)>1):
        return int(ts[0]) * 60 + float(ts[1])
    else:
        return float(ts[0])




def main():
    get_driver_info(44)
    get_session_info('Canada', 2023)
    get_position_info(9558)
    get_circuit_info('Canada')







if __name__ == '__main__':
    main()
