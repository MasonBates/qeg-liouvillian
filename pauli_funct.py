import codecs
import pauliarray as pa
import numpy as np
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import matplotlib.pyplot as plt
import pauliarray.pauli.operator as op
import pauliarray.pauli.pauli_array as pa
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.special import comb
from scipy.special import j0
from scipy.optimize import curve_fit
from scipy.special import j0
import numpy as np



def generate_dipole_hamiltonian(n, neighbor_list=None):
    
    """ Creates an operator object that behaves as the dipole Hamiltonian"""

    edges = neighbor_list if neighbor_list is not None else [(k,(k+1)% n) for k in range(n)]
    # Unless we intentionally create a "neighbour_list" (for the edges), it will default to creating the nearest neighbour in a ring interaction as before 

    ising_z = np.zeros((len(edges), n), dtype=bool)
    ising_x = np.zeros((len(edges), n), dtype=bool)
    ff1_z = np.zeros((len(edges), n), dtype=bool)
    ff1_x = np.zeros((len(edges), n), dtype=bool)
    ff2_z = np.zeros((len(edges), n), dtype=bool)
    ff2_x = np.zeros((len(edges), n), dtype=bool)

    for idx, edge in enumerate(edges):
        # ZZ term
        ising_z[idx, edge] = True
        # XX term
        ff1_x[idx, edge] = True
        # YY term
        ff2_z[idx, edge] = True
        ff2_x[idx, edge] = True
        
    ising_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(ising_z, ising_x), 1)
    ff1_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(ff1_z, ff1_x), -0.5)
    ff2_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(ff2_z, ff2_x), -0.5)

    return ising_hamiltonian.__add__(ff1_hamiltonian).__add__(ff2_hamiltonian)



def generate_dq_hamiltonian(n, neighbor_list=None):
    
    """Creates an operator object that behaves as the double quantum (DQ) hamiltonian"""

    edges = neighbor_list if neighbor_list is not None else [(k,(k+1)% n) for k in range(n)]

    dq1_z = np.zeros((len(edges), n), dtype=bool)
    dq1_x = np.zeros((len(edges), n), dtype=bool)
    dq2_z = np.zeros((len(edges), n), dtype=bool)
    dq2_x = np.zeros((len(edges), n), dtype=bool)

    for idx, edge in enumerate(edges):
        # XX term
        dq1_x[idx, edge] = True
        # YY term
        dq2_z[idx, edge] = True
        dq2_x[idx, edge] = True

    dq1_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(dq1_z, dq1_x), 1)
    dq2_hamiltonian = op.Operator.from_paulis_and_weights(pa.PauliArray(dq2_z, dq2_x), -1)
    return dq1_hamiltonian.__add__(dq2_hamiltonian)



def generate_louivillian_graph(seed_label, ham, depth, corr_cutoff=4, max_nodes=None, directed=True, decay_factor=1.0):
    """
    Generate a rustworkx PyGraph representing successive commutators
    with `ham` starting from `seed_label` up to `depth` layers.

    Parameters
    - seed_label: str, Pauli string like 'IIIIIZIIIII'
    - ham: pauliarray.pauli.operator.Operator, the Hamiltonian/operator to commute with
    - depth: int, number of commutator layers to generate (depth=0 => only seed node)
    - max_nodes: int or None, optional cap on total nodes (prevents explosion)

    Returns
    - (graph, label_to_idx) tuple where graph is rx.PyGraph and label_to_idx is a dict
      mapping pauli-label -> node index in the graph.
    """
    g = rx.PyDiGraph(multigraph=False) if directed else rx.PyGraph(multigraph=False)        # by default it should be directed; the Liouvillian is generally not Hermitian (works backwards with the same weight)
    label_to_idx = {}

    root_idx = g.add_node(seed_label)
    label_to_idx[seed_label] = root_idx
    # this is the starting (t=0) Pauli string. Over time the initial Pauli string must evolve per the Liouville equation, and be decomposed into more Pauli strings per eqn 14 of White's paper
    
    current_layer = [root_idx]      # initialise the list through which we will iterate
    
    # Negative depth is meaningless. Depth of 0 is just the initial "seed" node
    if depth <= 0:
        return g, label_to_idx

    # # NOTE: The loop that begins constructing the graph (this is where the fun begins)

    for _layer in range(depth):     # the number of times to perform the commutator

        print("\033[31m=======================\033[0m")
        print(f"\033[31mLayer: {_layer}\n\033[0m")
        
        next_layer = []
        
        for node_idx in current_layer:
            label = g.get_node_data(node_idx) 
            print(f"label: {label}\n")
            
            # build operator from the label and commute
            pauli_arr = pa.PauliArray.from_labels(label)
            node_op = op.Operator.from_paulis_and_weights(pauli_arr, 1)     
            
            # The Liouvillian operation: commutation of the hamiltonian with the previous node to create new nodes
            o_next = op.commutator(ham, node_op)
            o_next.combine_repeated_terms(True) 
            print(f"o_next: {o_next.inspect()}\n")

            for p, w in zip(o_next.paulis, o_next.weights):

                # print(f"p: {p}")
                lab = p.to_labels()[0]                  # this will access the individual Pauli strings from the operator (sum of Pauli strings)
                lab_strip = lab.replace('I', '')        # The I's do nothing, so just replace them with a blank
                # print(f"lab_strip: {lab_strip}")
                
                # print(f"w: {w}")    
                w = np.real(1j * w)  # Liouvillian commutator factor. Remember that the Liouville equation has an imaginary unit premultiplied to the commutator
                # So what we want is to take the individual Pauli strings inside the new operator and separate them out into different nodes, since each node is one Pauli string
                
                if len(lab_strip) > corr_cutoff:
                    # connect to itself with "imaginary" self-energy, so the effective term will be d(A)/dt = -1 * decay_factor * (A), where A is an operator
                    # NOTE: to my understanding, if there's "too many" Paulis in a Pauli string, then it will need to die off in time, hence connecting to itself and decaying its own existence
                    g.add_edge(node_idx, node_idx, -1 * decay_factor)
                    continue

                if lab in label_to_idx:
                    # node already exists, get its index, and then connect to that already existing node rather than incorrectly making a new node for the same Pauli string
                    new_idx = label_to_idx[lab]

                else:
                    # new node, add to graph and enforce optional node cap
                    if max_nodes is not None and len(label_to_idx) >= max_nodes:
                        continue

                    new_idx = g.add_node(lab)
                    next_layer.append(new_idx)      # create the next set of nodes to commute with, which is basically just the nodes that compose the original result of the commutation
                    label_to_idx[lab] = new_idx

                # add edge from initial to target pauli, with the relevant matrix element
                g.add_edge(node_idx, new_idx, w)

        current_layer = next_layer

        if not current_layer:
            break

    return g, label_to_idx



def node_attr(node):
    s = str(node)

    # parts = [f"{ch}" for ch in s if ch != "I"]        # Doesn't indicate the site number
    parts = []
    for chno, ch in enumerate(s):
        if ch != "I":
            if chno < 10:
                parts.append(codecs.decode(fr"{ch}\u208{chno}",'unicode_escape'))        # count from qubit 0 to qubit 10 (add +1 if qubit 1 to 11)
            elif 10 <= chno < 100:
                parts.append(codecs.decode(fr"{ch}\u208{chno//10}\u208{chno%10}",'unicode_escape'))
        

    count = len(parts)
    if count == 0:
        return {"label": s}
    # base blue intensity for one non-'I' char, reduce for each additional char
    base = 255
    decrement = 50
    blue = max(30, base - (count - 1) * decrement)
    hex_color = f"#{0:02x}{0:02x}{blue:02x}"
    # return early so the final return isn't reached
    return {"label": " ".join(parts), "color": "blue" if count == 1 else "red", "fillcolor": "blue" if count == 1 else "red", "style": "filled", "fontcolor": "white", "shape": "ellipse", "fontsize": "9", "width": "0.7", "height": "0.7", "fixedsize": "true"}
    #return {"label": " ".join(parts) if parts else s}
    


def edge_attr(edge):
    w = edge
    return {"label": f"{np.real(w):.2f}"}



def exp_action_via_eig(mat, vec, t):        
    """
    Compute exp(mat * t) @ vec using eigendecomposition with a scipy fallback for ill-conditioned matrices.     

    mat: (n,n) ndarray
    vec: (n,) ndarray
    t: scalar
    """

    mat = np.asarray(mat)
    vec = np.asarray(vec)
    assert mat.shape[0] == mat.shape[1] == vec.shape[0]     # Assert that the matrix shape must be square (rows = columns) and the vector's dimension is compatible with the matrix
    
    vals, vecs = np.linalg.eig(mat)     # eigenvalues and eigenvectors of the matrix we've inputted
    
    # if eigenvector matrix is ill-conditioned (large condition number), fallback to scipy.linalg.expm if available. Large condition number = sensitive to small change in input 
    if np.linalg.cond(vecs) > 1e12:         
        try:
            M = expm(mat * t)
            res = M.dot(vec)
            return np.real_if_close(res)
        except Exception:
            raise RuntimeError("Matrix appears nearly defective; install scipy or use a numerical integrator.")
    
    expD = np.diag(np.exp(vals * t))        # This is equivalent to the result of exponentiating the diagonal matrix of mat's eigenvalues (multiplied by some scalar t)

    M = vecs.dot(expD).dot(np.linalg.inv(vecs))     
    # Basically if M = PDP^-1 , then exp(Mt) = P exp(Dt) P^-1. Maths!

    return np.real_if_close(M.dot(vec))         # vec != vecs, don't confuse. 



def solve_chain_time_series(chain_mat, A0, times, max_step=.01):
    """
    Compute A(t) = exp(chain_mat * t) @ A0 for an array of times.
    
    chain_mat: (n,n) ndarray
    A0: (n,) ndarray initial condition
    times: 1D array-like of times
    Returns: (len(times), n) ndarray
    """
    times = np.asarray(times)
    A0 = np.asarray(A0)

    # in solve_ivp, what we are doing is that dy/dt = f(y,t) = chain_mat @ y
    def rhs(t, y):
        return chain_mat @ y

    # This scipy function solves an "initial value problem" of some functionm, performing a numerical integration  
    sol = solve_ivp(fun=rhs, t_span=(times[0], times[-1]), y0=A0, t_eval=times, method='RK45',max_step=max_step)      # RK45 = Runge-Kutta of order 5(4)
    # sol is a "solution" object, which has attributes sol.t and sol.y; sol.y are the solution values at the times the solution was evaluated (time steps given above)

    return sol.y.T      # transpose of sol.y



def pauli_to_mqc(p_string, basis="Z"):
    r"""
    Take a Pauli string and return a dictionary mapping MQC orders to their coefficients
    in the given basis. The MQC order of an operators Oq is defined in a basis of the 
    (collective) operator P = \sum_i \sigma_mu^{(i)} as [P, Oq ] = q Oq. When P is a 
    Pauli-Z, the MQC order can be computed by counting the number of sigma_+ operators minus
    the number of sigma_- operators in the Pauli string.
    """
    mqc = {}
    match basis:
        case "Z":
            num_x = p_string.count('X')
            num_y = p_string.count('Y')
            n = num_x + num_y
            for nminus in range(0, 2*n+1, 2):
                mqc[n-nminus] = (1j)**num_y * np.sum([comb(num_x, k)*comb(num_y, nminus//2-k)*((-1)**(nminus//2 - k) ) for k in range(0, nminus//2+1)])
        case "X":
            pass  # implement later
        case "Y":
            pass  # implement later
        case _:
            raise ValueError(f"Unknown basis: {basis}")
        
    return mqc



def compute_time_evolution(graph, A0, times, weight_fn=lambda x: x, max_step=0.01):
    """
    Compute the time evolution of a system represented by a Liouvillian graph.

    Parameters:
    - graph: rustworkx.PyDiGraph or rustworkx.PyGraph
        The Liouvillian graph representing the system.
    - A0: numpy.ndarray
        The initial state vector.
    - times: numpy.ndarray
        Array of time points for the evolution.
    - weight_fn: callable, optional
        A function to extract weights from graph edges. Defaults to identity.
    - max_step: float, optional
        Maximum step size for the numerical integrator.

    Returns:
    - A_t: numpy.ndarray
        The time-evolved state vector at each time point.
    """
    # Generate the adjacency matrix using rustworkx
    chain_mat = rx.adjacency_matrix(graph, weight_fn=weight_fn).T

    # Solve the time evolution using the solve_chain_time_series function
    A_t = solve_chain_time_series(chain_mat, A0, times, max_step=max_step)
    return A_t



def plot_mqc_from_label_map(label_map, A_t, times=None):
    """
    Generate an MQC plot from a label_map and time-evolved operator weights.

    Parameters:
    - label_map: dict, mapping of Pauli labels to indices in A_t
    - A_t: numpy.ndarray, time-evolved operator weights (shape: [len(times), n])
    - times: numpy.ndarray, array of time points

    Returns:
    - None, displays the MQC plot
    """
    # Compute MQC intensities
    mqc_intensity = {}
    if times is None:
        times = np.arange(A_t.shape[0])
    for label, idx in label_map.items():
        mqc_orders = pauli_to_mqc(label)
        for order, coeff in mqc_orders.items():
            if order not in mqc_intensity:
                mqc_intensity[order] = np.zeros(len(times), dtype=np.complex128)
            mqc_intensity[order] += coeff * A_t[:, idx]

    # Convert to intensity (magnitude squared)
    mqc_intensity = {order: np.abs(intensity)**2 for order, intensity in mqc_intensity.items()}

    # Create a 3D wire plot for MQC intensities
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    orders = sorted(mqc_intensity.keys())
    X, Y = np.meshgrid(times, orders)
    Z = np.array([mqc_intensity[order] for order in orders])

    for i in range(len(orders)):
        ax.plot(times, [orders[i]] * len(times), Z[i], label=f"Order {orders[i]}")

    ax.set_xlabel("Time")
    ax.set_ylabel("MQC Order")
    ax.set_zlabel("Intensity")
    ax.set_title("3D Wire Plot of MQC Intensities")
    plt.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
    # Compute the second moment of the MQC intensity (OTOC)
    otoc = np.zeros(len(times))
    for order, intensity in mqc_intensity.items():
        otoc += order**2 * intensity

    # Plot the OTOC as a function of time
    plt.figure(figsize=(6, 4))
    plt.plot(times, otoc, label="OTOC", color="purple", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("OTOC")
    plt.title("Out-of-Time-Ordered Correlator (OTOC) vs Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# Define the Bessel function model
def bessel_model(t, amplitude, frequency, phase):
    return amplitude * j0(2 * np.pi * frequency * t + phase)