import json
import requests
from bs4 import BeautifulSoup


def parse_political_views(url):
    '''
    Function that builds red and blue state lists from 2012 election
    INPUT: url to the wikipedia website used for scrapping 2012 election data
    OUPUT: red and blue state lists
    '''
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'html.parser')

    data_red = soup.find_all('tr', style="background:#FFB6B6")
    data_blue = soup.find_all('tr', style="background:#B0CEFF")

    red_states_td = [tag.find_all('td')[-1] for tag in data_red][2:]
    blue_states_td = [tag.find_all('td')[-1] for tag in data_blue][2:]

    red_states = [tag.text for tag in red_states_td][2:]
    blue_states = [tag.text for tag in red_states_td][2:]

    return red_states, blue_states

def red_dict(red_list):
    '''
    Function that builds red state dictionary
    INPUT: red state list
    OUPUT: red state dictionary
    '''
    red_dict = {}
    for state in red_list:
        if 'NE' in state:
            continue
        else:
            red_dict[state]='red'
    red_dict['NE']='red'
    return red_dict

def blue_dict(blue_list):
    '''
    Function that builds red state dictionary
    INPUT: blue state list
    OUPUT: blue state dictionary
    '''
    blue_dict = {}
    for state in blue_list:
        if 'ME' in state:
            continue
        else:
            blue_dict[state]='blue'
    blue_dict['ME']='blue'
    return blue_dict

def write_json(data, jsonfile):
    '''
    Function that creates a json dictionary file
    INPUT: dictionary and filename and path
    OUPUT: json file to the specified path
    '''
    with open(jsonfile, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":

    url = 'https://en.wikipedia.org/wiki/United_States_presidential_election,_2012#Results'

    red_states, blue_states = parse_political_views(url)

    state_dict = red_dict(red_states)
    state_dict.update(blue_dict(blue_states))

    write_json(state_dict, '../data/red_blue_states.json')
