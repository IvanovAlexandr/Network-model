import networkx as nx

import matplotlib.pyplot as plt
import random

from networkx.algorithms.flow import ford_fulkerson

# params
duplex = 0
half_duplex = 1
satellite = 2
virtual_chanel = 0
datagram = 1

matrix = []
node_n = 30
av_deg = 3.5
inf_packet_size = 1000
service_packet_size = 100
arr_weight = [2, 3, 6, 9, 12, 14, 16, 18, 20, 21, 25]
colors = ['red', 'green', 'blue']
n_satellite = 0
# channel capacity = def_capacity / weight
def_capacity = 100000000  # 100 Mbit/s

routing_tab = []
sending_tab = []


def average_deg():
    s = 0
    for i in range(node_n):
        s += sum(matrix[i])

    return float(s) / node_n


def generate_adjacency_matrix():
    for i in range(node_n):
        matrix.append([])
        for j in range(node_n):
            matrix[i].append(0)

    for i in range(node_n - 1):
        j = random.randint(i + 1, node_n - 1)
        matrix[i][j] = 1
        matrix[j][i] = matrix[i][j]

    deg = average_deg()
    while deg < av_deg:
        i = random.randint(0, node_n - 1)
        j = random.randint(0, node_n - 1)
        if i != j and matrix[i][j] == 0:
            matrix[i][j] = 1
            matrix[j][i] = 1
        deg = average_deg()


def create_network():
    global n_satellite
    G = nx.Graph()
    for i in range(node_n):
        for j in range(i, node_n):
            if i != j:
                if matrix[i][j] == 1:
                    if n_satellite < 2:
                        edge_type = random.randint(0, 2)
                        if edge_type == 2:
                            n_satellite += 1
                    else:
                        edge_type = random.randint(0, 1)

                    w = arr_weight[random.randint(0, 10)]
                    G.add_edge(i, j, weight=w, type=edge_type, err=random.uniform(0, 0.05),
                               capacity=def_capacity / (w * (edge_type + 1)))
    return G


def print_network(G):
    edge_colors = []
    for edge in G.edges():
        edge_colors.append(colors[G[edge[0]][edge[1]]['type']])
    pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx(G, pos, edge_color=edge_colors, width=1.5, with_labels=True, node_color="green", node_shape='*', node_size=1200)
    nx.draw_circular(G, edge_color=edge_colors, width=1.5, with_labels=True, node_color="green", node_size=300)
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    for label in edge_labels:
        edge_labels[label] = str(G[label[0]][label[1]]['weight']) + " (" \
                             + str(round(float(edge_labels[label]) * 100 / def_capacity, 2)) + " Mbit/s)"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4, font_size=8)
    plt.savefig("edge_colormap.png")
    plt.show()


def menu():
    print "NETWORK Params:"
    print "\tNode n :", node_n
    print "\tAverage degree :", av_deg
    print "\tWeights :", arr_weight
    print "\n\nMenu"
    print "1. Print Graph"
    print "2. Print routes table"
    print "3. Add/remove node"
    print "4. Add/remove edge"
    print "5. Testing"
    print "0. Exit"
    key = input("Enter: ")
    if key == 0:
        exit()
    elif key == 1:
        print_network(G)
        print "\n\n\n"
    elif key == 2:
        print_routing()
        print "\n\n\n"
    elif key == 3:
        menu_node()
        print "\n\n\n"
    elif key == 4:
        menu_edge()
        print "\n\n\n"
    elif key == 5:
        testing()
        print "\n\n\n"
    else:
        print "\n\n\n"
        print "Wrong command"


def menu_node():
    while True:
        print "\n\n\n"
        print "Node:"
        print "1. Add node"
        print "2. Remove node"
        print "0. Back"
        key = input("Enter: ")
        if key == 0:
            return
        elif key == 1:
            add_node()
        elif key == 2:
            remove_node()
        else:
            print "Wrong command"
    print "\n\n\n"


def menu_edge():
    while True:
        print "\n\n\n"
        print "Edge:"
        print "1. Add edge"
        print "2. Remove edge"
        print "0. Back"
        key = input("Enter: ")
        if key == 0:
            return
        elif key == 1:
            add_edge()
        elif key == 2:
            remove_edge()
        else:
            print "Wrong command"
    print "\n\n\n"


def add_node():
    global node_n
    global av_deg
    n = 0
    f = True
    while n in G.nodes():
        if f:
            n = input("Enter number node :")
            f = False
        else:
            print "Node with the number already exists"
            n = input("Enter another number :")
    G.add_node(n)
    node_n += 1
    av_deg = float(sum(G.degree().values())) / node_n
    print "Add node", n
    generate_routes()
    raw_input("Press enter...")


def remove_node():
    global node_n
    global av_deg
    n = None
    f = True
    while n not in G.nodes():
        if f:
            n = input("Enter number node :")
            f = False
        else:
            print "Node with the number doesn't exists"
            n = input("Enter another number :")
    print "Remove node", n, "? You are sure?"
    print "1. Yes"
    print "2. No"
    r = input()
    if r == 1:
        G.remove_node(n)
        node_n -= 1
        av_deg = float(sum(G.degree().values())) / node_n
        generate_routes()


def add_edge():
    global av_deg
    n1 = None
    n2 = None
    f = True
    weight = 0
    type = 0
    while n1 not in G.nodes():
        if f:
            n1 = input("Enter number node 1 :")
            f = False
        else:
            print "Node with this number doesn't exist"
            n1 = input("Enter another number node 1:")
    f = True
    while n2 not in G.nodes() or n2 == n1:
        if f:
            n2 = input("Enter number node 2 :")
            f = False
        else:
            if n2 == n1:
                print "Node with this number is node 1"
            else:
                print "Node with this number doesn't exist"
            n2 = input("Enter another number node 2:")

    dup = 0
    if (n1, n2) in G.edges() or (n2, n1) in G.edges():
        print "Such edge exists"
        dup = input("Duplicate? (1 -Yes, 2 - No)")

    if dup != 2:
        while True:
            print "Enter weight from this list", arr_weight
            weight = input()
            if weight in arr_weight:
                break
            else:
                print "Wrong weight"

        while True:
            print "Enter type :"
            print "0. Duplex"
            print "1. Half duplex"
            print "2. Satellite"
            type = input()
            if type in [duplex, half_duplex, satellite]:
                break
            else:
                print "Wrong type"

        G.add_edge(n1, n2, weight=weight, type=type, err=random.uniform(0, 0.15),
                   capacity=100 - (weight * (type + 1)))
        av_deg = float(sum(G.degree().values())) / node_n
    print "Add edge (", n1, ",", n2, "), weight =", weight, ", type =", type
    generate_routes()
    raw_input("Press enter...")


def remove_edge():
    global av_deg
    n1 = None
    n2 = None
    f = True
    while n1 not in G.nodes():
        if f:
            n1 = input("Enter number node 1 :")
            f = False
        else:
            print "Node with this number doesn't exist"
            n1 = input("Enter another number node 1:")
    f = True
    while n2 not in G.nodes() or n2 == n1:
        if f:
            n2 = input("Enter number node 2 :")
            f = False
        else:
            if n2 == n1:
                print "Node with this number is node 1"
            else:
                print "Node with this number doesn't exist"
            n2 = input("Enter another number node 2:")

    if (n1, n2) in G.edges():
        print "Remove edge (", n1, ",", n2, ")? You are sure?"
        print "1. Yes"
        print "2. No"
        r = input()
        if r == 1:
            G.remove_edge(n1, n2)
            av_deg = float(sum(G.degree().values())) / node_n
            generate_routes()
    elif (n2, n1) in G.edges():
        print "Remove edge (", n2, ",", n1, ")? You are sure?"
        print "1. Yes"
        print "2. No"
        r = input()
        if r == 1:
            G.remove_edge(n2, n1)
            av_deg = float(sum(G.degree().values())) / node_n
            generate_routes()
    else:
        print "This edge doesn't exist"
    raw_input("Press enter...")


def m_max(dict, node_stack):
    max_f = 0
    indexes = []
    for i in dict:
        if i not in node_stack:
            if dict[i] > max_f:
                max_f = dict[i]
                indexes = [i]
            elif dict[i] == max_f:
                indexes.append(i)

    return max_f, indexes


def routes(s, t):
    R = ford_fulkerson.ford_fulkerson(G, s, t)
    flow_value = R.graph['flow_value']
    flow_dict = R.graph['flow_dict']

    result_flow = []
    result_route = []
    visited = [t]

    for i in flow_dict[t]:
        if i != t:
            arr_index = []
            arr_flow = []
            node_stack = visited
            max_f, buf = m_max(flow_dict[t], node_stack)
            arr_index.append(buf)
            arr_flow.append(max_f)
            next_node = arr_index[len(arr_index) - 1].pop()
            node_stack = [t, next_node]
            visited.append(next_node)
            while True:
                if next_node == s:
                    break
                max_f, buf = m_max(flow_dict[next_node], node_stack)
                arr_index.append(buf)
                arr_flow.append(max_f)
                if len(arr_index[len(arr_index) - 1]) == 0:
                    break
                next_node = arr_index[len(arr_index) - 1].pop()
                node_stack.append(next_node)
            result_route.append(node_stack)
            if len(arr_flow) != 0:
                result_flow.append(min(arr_flow))

    return result_flow, result_route


def print_routing():
    print 'Source\t', 'Sink\t', 'Max flow (Mbit\s)\t', 'Route with min weight (max flow)\t', 'Min distance flow (Mbit\s)\t', \
        'Route with min distance\t'
    for row in routing_tab:
        str_min_weight = ""
        str_min_distance = ""
        str_min_distance_flow = str(row['min_dist_flow'])
        str_max_flow = str(row['max_flow'])
        for i in row['min_weight_route']:
            if str_min_weight != "":
                str_min_weight += "<-"
            str_min_weight += str(i)
        for i in range(len("Route with min weight (max flow)") - len(str_min_weight)):
            str_min_weight += " "
        for i in row['min_distance_route']:
            if str_min_distance != "":
                str_min_distance += "<-"
            str_min_distance += str(i)
        for i in range(len("Min distance flow (Mbit\s)") - len(str_min_distance_flow)):
            str_min_distance_flow += " "

        for i in range(len("Max flow (Mbit\s)") - len(str_max_flow)):
            str_max_flow += " "

        print row['source'], "\t\t", \
            row['sink'], "\t\t", \
            str_max_flow, "\t", \
            str_min_weight, "\t", \
            str_min_distance_flow, "\t", \
            str_min_distance
    raw_input("Press enter...")


def generate_routes():
    for source in G.nodes():
        for sink in G.nodes():
            if source != sink:
                flow, route = routes(source, sink)
                max_flow = 0
                index_max = 0
                for i in range(len(flow)):
                    if flow[i] > max_flow:
                        max_flow = flow[i]
                        index_max = i
                min_len = 0
                index_min_len = 0
                for i in range(len(route)):
                    l = len(route[i])
                    if l < min_len or min_len == 0:
                        if route[i][len(route[i]) - 1] == source:
                            min_len = l
                            index_min_len = i

                routing_tab.append({'source': source,
                                    'sink': sink,
                                    'arr_flow': flow,
                                    'arr_route': route,
                                    'max_flow': round(float(flow[index_max]) * 100 / def_capacity, 2),
                                    'min_weight_route': route[index_max],
                                    'min_dist_flow': round(float(flow[index_min_len]) * 100 / def_capacity, 2),
                                    'min_distance_route': route[index_min_len]})


def send_massage(type, source, sink, size):
    # if (source, sink) in G.edges() or (sink, source) in G.edges():
    # size in byte
    route = []
    flow = 0
    if type == virtual_chanel:
        for i in routing_tab:
            if i['source'] == source:
                if i['sink'] == sink:
                    route = i['min_weight_route']
                    flow = i['max_flow']
                    break

        n_inf_packet = size / inf_packet_size
        if size % inf_packet_size != 0:
            n_inf_packet += 1

        n_inf_packet *= len(route) - 1

        add = 0
        for i in range(n_inf_packet):
            a = route[0]
            for b in range(1, len(route)):

                x = random.random()
                if G[a][route[b]]['err'] > x:
                    add += 1
                a = route[b]

        n_inf_packet += add
        n_service_packet = n_inf_packet + 4 * (len(route) - 1)

        add = 0
        for i in range(n_service_packet):
            a = route[0]
            for b in range(1, len(route)):

                x = random.random()
                if G[a][route[b]]['err'] > x:
                    add += 1
                a = route[b]
        n_service_packet += add

        size_inf_part = n_inf_packet * inf_packet_size
        size_service_part = n_service_packet * service_packet_size
        time = round(((size_inf_part + size_service_part) / (flow * 1000000)) * 1000, 2)
    else:
        arr_route = []
        arr_flow = []
        queue_route = []
        for i in routing_tab:
            if i['source'] == source:
                if i['sink'] == sink:
                    arr_route = i['arr_route']
                    arr_flow = i['arr_flow']
                    break
        arr = {}
        for i in range(len(arr_flow)):
            arr[arr_flow[i]] = arr_route[i]
        arr = sorted(arr.items())
        if arr[0][0] == 0:
            arr.pop(0)
        routes_time_inf = []
        routes_time_service = []
        for i in arr:
            routes_time_inf.append(round((float(inf_packet_size) / i[0]) * 1000, 2))
            routes_time_service.append(round((float(service_packet_size) / i[0]) * 1000, 2))
            queue_route.append(0)
        n_inf_packet = size / inf_packet_size
        if size % inf_packet_size != 0:
            n_inf_packet += 1
        time = 0
        sum_n_packet = 0
        size_inf_part = 0
        size_service_part = 0
        add_inf = 0
        add_service = 0

        for packet in range(n_inf_packet):
            delay = []
            for i in range(len(queue_route)):
                delay.append((queue_route[i] + 1) * routes_time_inf[i])
            index_min_delay = len(delay) - 1
            min_delay = delay[index_min_delay]
            for i in range(len(delay)):
                if delay[i] <= min_delay:
                    min_delay = delay[i]
                    index_min_delay = i

            r = arr[index_min_delay][1]
            a = r[0]
            for b in range(1, len(r)):
                x1 = random.random()
                x2 = random.random()
                if G[a][r[b]]['err'] > x1:
                    add_inf += 1
                    add_service += 1
                if G[a][r[b]]['err'] > x2:
                    add_service += 1
                a = r[b]

            queue_route[index_min_delay] += 1
            k = (len(arr_route[index_min_delay]) - 1)
            sum_n_packet += k
            size_inf_part += inf_packet_size * k
            size_service_part += service_packet_size * k
            for i in range(len(delay)):
                delay[i] -= routes_time_inf[i]
            time = max(delay)

        n_inf_packet = sum_n_packet + add_inf
        n_service_packet = sum_n_packet + add_service

    sending_tab.append({'type': type,
                        'source': source,
                        'sink': sink,
                        'n_inf_packet': n_inf_packet,
                        'n_service_packet': n_service_packet,
                        'size_inf_part': size_inf_part,
                        'size_service_part': size_service_part,
                        'time': time,
                        'route': route})


def print_sending_tab():
    global sending_tab
    print 'Type\t', 'Source\t', 'Sink\t', 'Amount information packages\t', 'Amount service packages\t', \
        'Size information part\t', 'Size service part\t', 'Time\t\t', 'Route'
    for row in sending_tab:
        str_type = ""
        if row['type'] == 0:
            str_type = "VC"
        else:
            str_type = "DG"
        str_n_inf_packet = str(row['n_inf_packet'])
        str_n_service_packet = str(row['n_service_packet'])
        str_size_inf_part = str(row['size_inf_part'])
        str_size_service_part = str(row['size_service_part'])
        str_time = str(row['time'])
        str_route = ""
        for i in range(len('Amount information packages') - len(str_n_inf_packet)):
            str_n_inf_packet += " "
        for i in range(len('Amount service packages') - len(str_n_service_packet)):
            str_n_service_packet += " "
        for i in range(len('Size information part') - len(str_size_inf_part)):
            str_size_inf_part += " "
        for i in range(len('Size service part') - len(str_size_service_part)):
            str_size_service_part += " "
        for i in range(len('Time    ') - len(str_time)):
            str_time += " "
        for i in row['route']:
            if str_route != "":
                str_route += "<-"
            str_route += str(i)
        print str_type, "\t\t", \
            row['source'], "\t\t", \
            row['sink'], "\t\t", \
            str_n_inf_packet, \
            str_n_service_packet, \
            str_size_inf_part, "\t", \
            str_size_service_part, "\t", \
            str_time, "\t", \
            str_route
    raw_input("Press enter...")

    sending_tab = []


def testing():
    """
        virtual channel and datagram
    """
    print "virtual channel and datagram "
    for i in range(10):
        while True:
            n1 = random.randint(0, node_n - 1)
            n2 = random.randint(0, node_n - 1)
            if n1 != n2:
                break
        send_massage(virtual_chanel, n1, n2, 5000000)
        send_massage(datagram, n1, n2, 5000000)
    print_sending_tab()

    """
            different size
    """
    print "different size "

    for i in range(10):
        size = random.randint(1, 5000000)
        print size
        while True:
            n1 = random.randint(0, node_n - 1)
            n2 = random.randint(0, node_n - 1)
            if n1 != n2:
                break
        send_massage(virtual_chanel, n1, n2, size)
        send_massage(datagram, n1, n2, size)
    print_sending_tab()

    """
                different size of packet
        """
    print "different size of packet "

    global inf_packet_size
    while True:
        n1 = random.randint(0, node_n - 1)
        n2 = random.randint(0, node_n - 1)
        if n1 != n2:
            break
    for i in range(10):
        inf_packet_size += 1000
        send_massage(virtual_chanel, n1, n2, 5000000)
        send_massage(datagram, n1, n2, 5000000)
    inf_packet_size = 1000
    print_sending_tab()


if __name__ == '__main__':
    generate_adjacency_matrix()
    G = create_network()
    av_deg = float(sum(G.degree().values())) / node_n
    generate_routes()

    while True:
        menu()
