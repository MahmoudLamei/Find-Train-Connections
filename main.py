import math
import matplotlib.pyplot as plt
import pandas as pd
import heapq
import networkx as nx
import time
import pickle
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


start = time.time()


# Returns a DataFrame with the relevant columns only.
def read_data(file_path):
    schedule = pd.read_csv(file_path)
    relevant_columns = ['Train No.', 'islno', 'station Code', 'Arrival time', 'Departure time', 'Distance']
    schedule = schedule[relevant_columns]
    schedule['Train No.'] = schedule['Train No.'].str[1:-1]
    schedule['station Code'] = schedule['station Code'].str.strip()
    schedule['Arrival time'] = schedule['Arrival time'].str[1:-1]
    schedule['Departure time'] = schedule['Departure time'].str[1:-1]
    schedule['Arrival time'] = pd.to_datetime(schedule['Arrival time'], format='%H:%M:%S').dt.time
    schedule['Departure time'] = pd.to_datetime(schedule['Departure time'], format='%H:%M:%S').dt.time
    return schedule


def get_next_station(train_no, islno, schedule):
    # returns an empty dataframe if it's the last islno in the train it'll return an empty.
    next_station = schedule[(schedule['Train No.'] == train_no) & (schedule['islno'] == islno + 1)]
    return next_station


def create_graph(schedule):
    graph = nx.MultiDiGraph()
    for idx, row in schedule.iterrows():
        train_no, islno, station_code, distance, departure_time = \
            schedule.iloc[idx].loc[['Train No.', 'islno', 'station Code', 'Distance', 'Departure time']]
        next_station = get_next_station(train_no, islno, schedule)
        if not next_station.empty:
            next_station_code = next_station.iloc[0].loc['station Code']
            next_distance = next_station.Distance.iloc[0]
            arrival_time = next_station['Arrival time'].iloc[0]
            # Add to the edge: train number, distance from A to B, islno of A,
            # d_t: time the train moved from A (d_t from A), a_t: time the train arrived at B (a_t from B).
            graph.add_edge(station_code, next_station_code, Name=train_no, Distance=next_distance-distance,
                           islno=islno, Departure=departure_time, Arrival=arrival_time)

    return graph


def get_attrs(graph):
    train_no = nx.get_edge_attributes(graph, 'Name')
    dist = nx.get_edge_attributes(graph, 'Distance')
    arrival = nx.get_edge_attributes(graph, 'Arrival')
    departure = nx.get_edge_attributes(graph, 'Departure')
    return train_no, dist, arrival, departure


def time_difference(x, y):
    x_seconds = x.hour * 3600 + x.minute * 60 + x.second
    y_seconds = y.hour * 3600 + y.minute * 60 + y.second
    return x_seconds - y_seconds


def dijkstra(graph, start, goal, cost_function):
    # Initialize the distance dictionary and priority queue.
    is_arrivaltime = cost_function.split()[0] == 'arrivaltime'
    if not is_arrivaltime:
        cost = {node: float('inf') for node in graph.nodes}
        cost[start] = 0
        queue = [(0, start)]
    else:
        cost = {node: (float('inf'), float('inf')) for node in graph.nodes}
        cost[start] = (0, 0)
        queue = [((0,0), start)]

    previous = {node: (None, None, None, None) for node in graph.nodes}

    while queue:
        # Pop the node with the smallest distance from the queue.
        current_cost, current_node = heapq.heappop(queue)

        # Stop searching if the goal is reached.
        if current_node == goal:
            break

        # Skip if the current distance is greater than the stored distance.
        if is_arrivaltime and start != current_node and current_cost[0] > cost[current_node][0]:
            continue
        elif (not is_arrivaltime) and current_cost > cost[current_node]:
            continue
        elif is_arrivaltime and start == current_node and current_cost[0] > cost[current_node][0]:
            continue

        # Explore neighbors of the current node.
        for neighbor, edge_data in graph[current_node].items():
            islno = edge_data[0]['islno']
            train = edge_data[0]['Name']
            departure = edge_data[0]['Departure']
            arrival = edge_data[0]['Arrival']


            # Add 1 for each station the train passes by.
            match cost_function.split()[0]:
                # The number of times we enter a station by train.
                case 'stops':
                    new_cost = cost[current_node] + 1

                # The total amount of time spent in a moving train in seconds (for simplicity, we ignore the time
                # a train is in a train station and the time when trains are changed)
                case 'traveltime':
                    time_diff = time_difference(arrival, departure)
                    # Add 24hrs if the arrival is smaller because it will be a new day.
                    if arrival < departure:
                        time_diff += 86400
                    new_cost = cost[current_node] + time_diff

                # You can only buy the daily train ticket which costs 1.
                case 'price':
                    price = 0
                    if arrival < departure:
                        price += 1
                    elif previous[current_node][1] != train:
                        price += 1
                    else:
                        price += 0
                    new_cost = cost[current_node] + price

                # Start at a specific time and try to arrive as soon as possible by giving the time you arrived at with
                # the days added.
                case 'arrivaltime':
                    days = cost[current_node][1]
                    start_time = cost_function.split()[1]
                    if start == current_node and datetime.strptime(start_time, '%H:%M:%S').time().hour < departure.hour:
                        days += 1
                    time_diff = time_difference(arrival, departure)

                    if time_diff < 0:
                        days += 1
                        time_diff += 86400
                    new_cost = (cost[current_node][0] + time_diff, days)


            # Update the distance and add to the queue if shorter path found.
            if (not is_arrivaltime) and new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                previous[neighbor] = (current_node, train, islno, arrival)
                heapq.heappush(queue, (new_cost, neighbor))
            elif is_arrivaltime and new_cost[0] < cost[neighbor][0]:
                cost[neighbor] = new_cost
                previous[neighbor] = (current_node, train, islno, arrival)
                heapq.heappush(queue, (new_cost, neighbor))
    # Back-track the previous to get the path from start to goal.
    path = []
    current_node = goal
    while current_node is not None:
        temp = current_node
        current_node, train, islno, arrival = previous[current_node]
        path.append((temp, train, islno, arrival))

    path.reverse()

    if is_arrivaltime:
        last_arrival = path[-1][3]
        last_arrival_str = last_arrival.strftime("%H:%M:%S")
        last_arrival_with_day = str(cost[goal][1]).zfill(2) + ':' + last_arrival_str if cost[goal][1] != 0 \
            else last_arrival_str
        cost_res = last_arrival_with_day
    else:
        cost_res = cost[goal]

    return path, cost_res


# '01153:1->3;17001:4->5'
def format_result(path):
    res = ''
    last_train = None
    last_islno = None
    for i, (station, train, islno, arrival) in enumerate(path[1:]):
        # Train change.
        if train != last_train:
            # First train.
            if res == '':
                res += train + ' : ' + str(islno) + ' -> '
            # Any other train change.
            else:
                res += str(last_islno+1) + ' ; ' + train + ' : ' + str(islno) + ' -> '

        last_islno = islno
        last_train = train
        # For last connection.
        if i+1 == len(path)-1:
            res += str(last_islno+1)
    return res


def test():
    result_df = pd.DataFrame(columns=["ProblemNo", "Connection", "Cost"])
    # wrong_answers = ''
    full_graph = pickle.load(open('schedule_graph.pickle', 'rb'))
    mini_graph = pickle.load(open('mini-schedule_graph.pickle', 'rb'))

    # example_problems = pd.read_csv('example-problems.csv')
    # example_solutions = pd.read_csv('example-solutions.csv')

    problems = pd.read_csv('problems.csv')

    for i, row in problems.iterrows():
        # problem_solution = ''
        problem_num = row['ProblemNo']
        from_station = row['FromStation']
        to_station = row['ToStation']
        schedule = row['Schedule']
        cost_function = row['CostFunction']

        # sol_problem_num, sol_connection, sol_cost = example_solutions.iloc[i][['ProblemNo', 'Connection', 'Cost']]
        # solution = str(sol_problem_num) + ',' + str(sol_connection) + ',' + str(sol_cost)

        graph = full_graph if schedule == 'schedule.csv' else mini_graph
        path, cost = dijkstra(graph, from_station, to_station, cost_function)
        res = format_result(path)
        # problem_solution = str(problem_num) + ',' + res + ',' + str(cost)

        # if problem_solution != solution:
        #     print('Mine: ' + problem_solution + '\nReal: ' + solution + '\n')

        result_df.loc[len(result_df)] = [problem_num, res, cost]
        result_df.to_csv('solutions.csv', index=False)


# graph = pickle.load(open('schedule_graph.pickle', 'rb'))
# path, cost = dijkstra(graph, 'SYM', 'NRT', 'price')
# print(path, "\nCost: ", cost)
# print(format_result(path))
schedule = read_data('mini-schedule.csv')
df = schedule.head(6)
graph = create_graph(df)
g = nx.path_graph(graph)
nx.draw(g, with_labels=True)
plt.show()

# test()

end = time.time()
print("\nRuntime: ", end - start)
