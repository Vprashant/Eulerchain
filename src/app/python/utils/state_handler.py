"""
filename: state_handler.py
Author: Prashant Verma
email: prashantv@sabic.com
"""

from typing import Optional, Dict, List

class StateHandler:
    def __init__(self):
        self.states = {}

    def add_state(self, state_id, state_name, status=True, query='', response='', **kwargs):
        self.states[state_id] = {
            'state_id': state_id,
            'state_name': state_name,
            'status': status,
            'query': query,
            'response': response,
            'optional_params': {
                'score': [kwargs.get('score', [])],  # [kwargs.get('score')] if kwargs.get('score') is not None else [],
                'iterations': kwargs.get('iterations'),
                'data': kwargs.get('data')
            }
        }
        print(f"Added state: {self.states[state_id]}") 

    def get_state_count(self):
        return len(self.states)

    def update_state(self, state_id=None, state_name=None, status=None, query=None, response=None, optional_params=None):
        state = None
        if state_id is not None:
            state = self.states.get(state_id)
        elif state_name is not None:
            state = self.get_state_by_name(state_name)

        if state is None:
            raise ValueError(f"State with ID '{state_id}' or name '{state_name}' does not exist.")

        if status is not None:
            state['status'] = status
        if query is not None:
            state['query'] = query
        if response is not None:
            state['response'] = response
        if optional_params:
            if 'score' in optional_params:
                state['optional_params']['score'].append(optional_params['score'])
            if 'iterations' in optional_params:
                state['optional_params']['iterations'] = optional_params['iterations']
            if 'data' in optional_params:
                state['optional_params']['data'] = optional_params['data']

    def get_state(self, state_id):
        if state_id in self.states:
            return self.states[state_id]
        else:
            raise ValueError(f"State with ID '{state_id}' does not exist.")
        
    def get_state_by_name(self, state_name):
        for state in self.states.values():
            if state['state_name'] == state_name:
                return state
        raise ValueError(f"State with Name '{state_name}' does not exist.")

    def remove_state(self, state_id):
        if state_id in self.states:
            del self.states[state_id]
        else:
            raise ValueError(f"State with ID '{state_id}' does not exist.")
        
    def get_all_states(self):
        return [{key:value for key, value in val.items() if key in ["state_name"]} for val in self.states.values()]
    
    def get_state_performance_stats(self):
        return [{k: v for k, v in val.items() if k in ["state_id", "state_name", "status", "optional_params"]} for val in self.states.values()]



















