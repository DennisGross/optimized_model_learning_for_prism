import os
import stormpy
import pandas as pd
import numpy as np
from scipy import spatial
import random
import itertools
import ast
from sklearn.metrics import pairwise_distances
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import collections
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from itertools import product

PRISM_FILES_PATH = 'prism_files/'
PRISM_NAME = 'complex_drone'
GROUND_TRUTH_PATH = os.path.join(PRISM_FILES_PATH, f'{PRISM_NAME}.prism')
OUTPUT_PRISM_FILE_PATH = os.path.join(PRISM_FILES_PATH, f'{PRISM_NAME}_compressed.prism')

def build_model(file_path):
    prism_program = stormpy.parse_prism_program(file_path)

    options = stormpy.BuilderOptions()
    options.set_build_choice_labels()
    options.set_build_state_valuations()

    model = stormpy.build_sparse_model_with_options(prism_program, options)
    return model

class StateVariableInvestigator:

    def __init__(self):
        self.features = {}

    def analyze_state(self, state_str):
        state_str = state_str.replace("(","").replace(")","")
        for feature_assignment in state_str.split(","):
            feature_assignment = feature_assignment.strip()
            feature, value = feature_assignment.split("=")
            if feature not in self.features:
                self.features[feature] = {}
                self.features[feature]["max"] = int(value)
                self.features[feature]["min"] = int(value)
                self.features[feature]["init"] = int(value)
            else:
                if self.features[feature]["max"] < int(value):
                    self.features[feature]["max"] = int(value)
                if self.features[feature]["min"] > int(value):
                    self.features[feature]["min"] = int(value)

    def print_feature_domains(self):
        for feature in self.features.items():
            print(f"{feature}")

    def get_feature_max_domains(self):
        domains = {}
        for feature in self.features.keys():
            domains[feature] = self.features[feature]["max"]+1
        return domains



def state_to_str(model, state):
    raw_state_str = model.state_valuations.get_string(state)
    # [h=1 & speed=0       & energy=0]
    state_str = "("
    for part in raw_state_str.split("&"):
        part = part.replace("[","").replace("]","")
        old_part = part
        if part.startswith("!"):
            part = part.strip()[1:] + "=0"
        if part.find("=") == -1:
            part = part.strip() + "=1"
        state_str += f"{part.strip()},"
    state_str = state_str[:-1] +")"
    return state_str


def action_to_str(model, action):
    return list(model.choice_labeling.get_labels_of_choice(action.id))[0]


def resolve_state_for_comparison(state):
    # [h=1 & speed=0       & energy=0]
    state_str = "("
    for part in state.split("&"):
        if part.startswith("!"):
            part = part.strip()[1:] + "=0"
        if part.find("=") == -1:
            part = part.strip() + "=1"
        part = part.replace("[","").replace("]","")
        state_str += f"{part.strip()},"
    state_str = state_str[:-1] +")"
    return state_str

def compare_models(model1, model2):
    l1_norm = 0  # Initialize L1 norm to 0
    if len(model1.states) != len(model2.states):
        print("Number of states is different", len(model1.states), len(model2.states))
    if model1.nr_transitions != model2.nr_transitions:
        print("Number of transitions is different", model1.nr_transitions, model2.nr_transitions)
    for state1, state2 in product(model1.states, model2.states):
        state1_str = resolve_state_for_comparison(model1.state_valuations.get_string(state1))
        state2_str = resolve_state_for_comparison(model2.state_valuations.get_string(state2))
        if resolve_state_for_comparison(state1_str) == resolve_state_for_comparison(state2_str):
            for action1, action2 in product(state1.actions, state2.actions):
                #print(action1, action2)
                action_name1 = str(list(model1.choice_labeling.get_labels_of_choice(action1.id))[0])
                actiion_name2 = str(list(model2.choice_labeling.get_labels_of_choice(action2.id))[0])
                if action_name1 == actiion_name2:
                    for transition1, transition2 in zip(action1.transitions, action2.transitions):
                        difference = abs(transition1.value() - transition2.value())
                        l1_norm += difference  # Update the L1 norm

                        if difference > 0:  # If there's a difference between the transition probabilities
                            print("!!!!!!!!!!!!!!!!!")
                            print("#1 From state {} by action {}, with probability {}, go to state {}".format(state1_str, action_name1, transition1.value(), transition1.column))
                            print("#2 From state {} by action {}, with probability {}, go to state {}".format(state2_str, actiion_name2, transition2.value(), transition2.column))

    print("L1 norm between the models:", l1_norm)
    return l1_norm

m_state_variable_investigator = StateVariableInvestigator()

prism_program = stormpy.parse_prism_program(GROUND_TRUTH_PATH)

options = stormpy.BuilderOptions()
options.set_build_choice_labels()
options.set_build_state_valuations()

model = stormpy.build_sparse_model_with_options(prism_program, options)

NR_STATES = len(model.states)
all_states = {'states': []}
for state in model.states:
    print(state, NR_STATES)
    state_str = state_to_str(model, state)
    m_state_variable_investigator.analyze_state(state_str)
    state_dir = {state_str : {}}
    for action in state.actions:
        action_str = action_to_str(model, action)
        state_dir[state_str][action_str] = {}
        for transition in action.transitions:
            next_state_id = transition.column
            next_state_str = state_to_str(model, next_state_id)
            state_dir[state_str][action_str][next_state_str] = transition.value()
        all_states['states'].append(state_dir)





# dictionary to json file
with open('data.json', 'w') as outfile:
    json.dump(all_states, outfile)



# Read JSON
with open('data.json') as json_file:
    data_str = json_file.read()


# Parse JSON
data = json.loads(data_str)

# Transform data
frames = []
for state_dict in data["states"]:
    state = list(state_dict.keys())[0]
    for action, next_states in state_dict[state].items():
        row_data = {"state": state, "action": action}
        row_data.update(next_states)
        frames.append(pd.DataFrame([row_data]))

# Concatenate frames
df = pd.concat(frames, ignore_index=True)

# Nan to zero
df = df.fillna(0)

for column in df.columns[2:]:
    m_state_variable_investigator.analyze_state(column)

m_state_variable_investigator.print_feature_domains()
# Display
print(df)

total_occurrences = df.groupby(["state", "action"]).sum()
row_sums = total_occurrences.sum(axis=1)
state_action_probs = total_occurrences.div(row_sums, axis=0)
print(state_action_probs)

def generate_constraints(variables):
    constraints=[]
    for v in variables:
        c=  str(v)+'+0'
        constraints.append(c)
    for v,dom in variables.items():
        for i in range(1,dom):

                c1=  str(v)+'-'+str(i)
                c2=  str(v)+'+'+str(i)
                c3=  str(v)+'*'+str(i)

                constraints.append(c1)
                constraints.append(c2)
    return preprocess_constraints(constraints)
def preprocess_constraints(constraints):
    """Convert constraints into a more efficient format."""
    processed = []
    for constraint in constraints:
        var, op_val = constraint.split('-') if '-' in constraint else constraint.split('+')
        op = '-' if '-' in constraint else '+'
        val = int(op_val)
        processed.append((var, op, val))
    return processed

def get_probabilities(state,constraint,future):
    for f in future:
        if '-' in constraint:
            l = int(constraint.split('-')[1])
            x1=int(state.split('=')[1].replace(')',''))
            x2=int(f[0].split('=')[1].replace(')',''))

            if x2 == (x1 - l):
                return (f[1],constraint)
        elif '+' in constraint:
            l = int(constraint.split('+')[1])
            x1=int(state.split('=')[1].replace(')',''))
            x2=int(f[0].split('=')[1].replace(')',''))
            if x2 == (x1 + l):
                return (f[1],constraint)

def checker(B,A):

    """Optimized checker function."""
    left_variables = A[0]
    right_variables = A[1]

    for right_variable in right_variables:
        for b in B:
                var, op, val=b

                if op == '-':
                    if right_variable[var] == (left_variables[var] - val):
                        return True
                if op == '+':
                    if right_variable[var] == (left_variables[var] + val):
                        return True
                if op == '*':
                        if right_variable[var] == (left_variables[var] * val):
                            return True

    return False
def preprocess_constraints(constraints):
    """Convert constraints into a more efficient format."""
    processed = []
    for constraint in constraints:
        var, op_val = constraint.split('-') if '-' in constraint else constraint.split('+')
        op = '-' if '-' in constraint else '+'
        val = int(op_val)
        processed.append((var, op, val))
    return processed






def create_head_expression(tuples_list_):
    expressions = []

    for tuples_list in tuples_list_:
        t= tuples_list.split(',')
        t=[e.replace('(','') for e in t]
        t=[e.replace(')','') for e in t]

        if len(set(t)) >= 1:  # All elements are the same
            expressions.append(' & '.join(t))


    return ' | '.join( expr for expr in expressions)


def merge(properties):
    grouped_vars = defaultdict(list)

    for var, csts in properties.items():
        # Convert the list of constraints to a tuple so it can be used as a dict key
        key = tuple(sorted(csts))

        grouped_vars[key].append(var)
    final=defaultdict(list)
    for k,v in grouped_vars.items():
        expressions=create_head_expression(v)
        final[k].append(expressions)
    return final


def split(length):
    return length // 2

def get(set_, start, end):
    return set_[start-1:end]

def PRISM_ACQ(B,A):

        return QX_incremental_batch( B,A)

def QX_incremental_batch(B, A):
    """
    Adapted QuickXplain to find commands that satisfy the constraints.
    """
    if not B:
        return []

    if len(B) == 1:
        return [B] if checker(B, A) else []
    k = len(B) // 2
    first_half = B[:k]
    second_half = B[k:]
    #random.shuffle(first_half)
    #random.shuffle(second_half)

    satisfying_commands = []

    if checker(first_half, A) :
        satisfying_commands.extend(QX_incremental_batch(first_half, A))

    # Check the second half
    if checker(second_half, A):
        satisfying_commands.extend(QX_incremental_batch(second_half, A))

    return satisfying_commands

def linear_check(constraints, command):
    """
    Check constraints linearly for a given command.

    Parameters:
    - constraints: a list of constraints.
    - command: a command to be checked against the constraints.

    Returns:
    - A list of constraints that are satisfied by the command.
    """
    satisfied_constraints = []

    for constraint in constraints:
        if checker([constraint], command):
            satisfied_constraints.append(constraint)

    return satisfied_constraints

variables_and_domains= m_state_variable_investigator.get_feature_max_domains()

constraints=generate_constraints(variables_and_domains)



import time
import random
CommandsPerAction={}
CommandsPerAction1={}

t=0
t1=0
for idx,group in state_action_probs.groupby(['action']):
    properties={}
    properties1={}

    for i,g in group.iterrows():

            subsets=[]

            results=[]
            columns=list(group.columns)
            #random.shuffle(constraints)

            X={f.split('=')[0].replace('(',''):int(f.split('=')[1].replace(')','')) for f in i[0].split(',')}
            Y=[]
            for c in columns:
                if g.loc[c]>0:
                    Y.append({f.split('=')[0].replace('(',''):int(f.split('=')[1].replace(')','')) for f in c.split(',')})

            A=[X,Y]
            probabilities=[]
            for y in Y:
                column=[]
                for k,v in y.items():
                    column.append(str(k)+'='+str(v))
                column='('+','.join(column)+')'
                probabilities.append(g[column])
            start=time.time()
            propertie=PRISM_ACQ(constraints,A)
            end=time.time()
            t+=(end-start)
            start=time.time()

            propertie1=linear_check(constraints,A)
            end=time.time()
            t1+=(end-start)
            properties[i[0]]=[str([v,k]) for k, v in zip(propertie,probabilities)]
            properties1[i[0]]=[str([v,k]) for k, v in zip(propertie1,probabilities)]

    merged_properties=merge(properties)
    merged_properties1=merge(properties1)
    print(t)
    print(t1)

    CommandsPerAction[group.index[0][1]]=merged_properties
    CommandsPerAction1[group.index[0][1]]=merged_properties1

def parse_transition(transition):
    parts = transition.split(' [')
    prob = float(parts[0].replace("[", "").replace(",", "").strip())
    update = parts[1].replace("[", "").replace(",", "").replace("]", "").replace("'", "").replace("(", "").replace(")", "").strip()
    update = update.replace(" ", "")
    update_parts = update.split("+")
    if len(update_parts) == 1:
        update_parts = update.split("-")
    update_parts[0] = update_parts[0] + "'"
    # concate update parts
    update = "(" + update_parts[0] + "=" + update + ")"
    print(update)
    return prob, update


# h' = h + 1
# h+1
def extract_commands(CommandsPerAction):
    all_command_lines = []
    for action, _ in CommandsPerAction.items():
        #print(CommandsPerAction[key])
        for transitions,v in CommandsPerAction[action].items():
            guard = v[0]
            print("==================")
            command_line = f"[{action}] "
            command_line += f"{guard} -> "
            for transition in transitions:
                prob, update = parse_transition(transition)
                command_line += f"{prob} : {update} + "
            command_line = command_line[:-3] + ";"
            print(command_line)
            all_command_lines.append(command_line)
        print('------------------')
    return all_command_lines

#print(CommandsPerAction)
all_command_lines = extract_commands(CommandsPerAction)

f = open(OUTPUT_PRISM_FILE_PATH, "w")
f.write("mdp\n")
f.write("module compressed_env\n")
for feature_name, feature_values in m_state_variable_investigator.features.items():
    f.write(f'{feature_name} : [{feature_values["min"]}..{feature_values["max"]}] init {feature_values["init"]};\n')

for command_line in all_command_lines:
    f.write(command_line + "\n")

f.write("endmodule")
f.close()






model1 = build_model(GROUND_TRUTH_PATH)
model2 = build_model(OUTPUT_PRISM_FILE_PATH)

compare_models(model1, model2)
