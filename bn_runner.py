#!/usr/bin/env python3
"""
bn_tunner_solution.py — Instructor solution for bn_runner.py
Implements enumeration-based inference over Bayes Nets using the same
file format and I/O behavior as the student runner.

Usage (same as runner):
    python bn_tunner_solution.py burglary.txt
    python bn_tunner_solution.py sprinklers.txt


Authors: S. El Alaoui and ChatGPT 5

"""
import sys
import itertools

# --------------------------
# Data structures
# --------------------------

class BayesNet:
    def __init__(self):
        # var -> list of domain values in declared order
        self.domains = {}
        # var -> list of parent var names in order as declared
        self.parents = {}
        # var -> dict: key = tuple(parent_values_in_order),
        #               value = dict { var_value -> prob }
        self.cpt = {}
        # a fixed topological order (parents come before children)
        self.order = []
        # The list of root nodes
        self.roots = set()

    def values(self, var):
        return self.domains[var]

    def prob(self, var, value, assignment):
        """Return P(var=value | parents=assignment[parents])."""
        par = self.parents.get(var, [])
        key = tuple(assignment[p] for p in par) if par else ()
        return self.cpt[var][key][value]

# --------------------------
# Parsing
# --------------------------

def parse_bn(path):
    """Parse a BN file in the format used by burglary.txt and sprinklers.txt.
    Returns a BayesNet object with domains, parents, CPTs filled.
    """
    net = BayesNet()
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("//")]

    # Split into three sections
    if "# Parents" not in lines or "# Tables" not in lines:
        sys.exit("Input file missing '# Parents' or '# Tables' sections.")
    i_parents = lines.index("# Parents")
    i_tables = lines.index("# Tables")

    var_lines = lines[:i_parents]
    parent_lines = lines[i_parents + 1: i_tables]
    table_lines = lines[i_tables + 1:]

    # 1) Variables and domains
    for ln in var_lines:
        parts = ln.split()
        var, vals = parts[0], parts[1:]
        if not vals:
            sys.exit(f"Variable '{var}' missing domain values.")
        net.domains[var] = vals

    # 2) Parent structure
    for ln in parent_lines:
        parts = ln.split()
        var, pars = parts[0], parts[1:]
        net.parents[var] = pars
        net.cpt[var] = {}

    net.roots = set(net.domains.keys()) - set(net.parents.keys())
    for var in net.roots:
        net.cpt[var] = {}

    # 3) Tables (variable-labeled blocks)
    idx = 0
    while idx < len(table_lines):
        var = table_lines[idx].split()[0]
        if var not in net.domains:
            sys.exit(f"Unknown variable in # Tables: {var}")
        idx += 1

        par = net.parents.get(var, [])
        dom = net.domains[var]
        d = len(dom)

        # number of rows determined by parent domain sizes
        num_rows = 1
        for p in par:
            num_rows *= len(net.domains[p])

        for _ in range(num_rows):
            row = table_lines[idx].split()
            idx += 1
            if par:
                if len(row) < len(par) + (d - 1):
                    sys.exit(f"Bad CPT row for {var}")
                par_vals = tuple(row[:len(par)])
                prob_parts = row[len(par):]
            else:
                par_vals = ()
                prob_parts = row

            provided = [float(x) for x in prob_parts]
            if len(provided) != (d - 1):
                sys.exit(
                    f"CPT for {var} should provide {d-1} prob(s) per row "
                    f"(got {len(provided)})."
                )
            last = 1.0 - sum(provided)
            if last < 0 and last > -1e-12:
                last = 0.0

            full = provided + [last]
            net.cpt[var][par_vals] = {dom[i]: full[i] for i in range(d)}

    net.order = topo_order(net)
    return net


def topo_order(net: BayesNet):
    indeg = {v: 0 for v in net.domains}
    for v in net.domains:
        for p in net.parents.get(v, []):
            indeg[v] += 1
    Q = [v for v in net.domains if indeg[v] == 0]
    order = []
    while Q:
        v = Q.pop(0)
        order.append(v)
        for w in net.domains:
            if v in net.parents.get(w, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    Q.append(w)
    if len(order) != len(net.domains):
        sys.exit("Graph is not a DAG.")
    return order

# --------------------------
# SOLUTION IMPLEMENTATIONS
# --------------------------

def joint_probability(net: BayesNet, assignment: dict) -> float:
    """Compute and return the joint probability of a COMPLETE assignment.

    Args:
        net: BayesNet
        assignment: dict mapping every variable -> one of its domain values.

    Returns:
        Product over variables X of P(X = assignment[X] | Parents(X) = assignment[Parents(X)]).
    """
    raise NotImplementedError


def update(distribution: dict, value, p: float) -> None:
    """Add probability mass 'p' into the running distribution for a query variable.

    Args:
        distribution: dict value -> mass
        value: a domain value of the query variable
        p: nonnegative probability mass to add
    """
    raise NotImplementedError

def normalize(distribution: dict) -> None:
    """Normalize a one-dimensional distribution so its values sum to 1.

    Args:
        distribution: dict value -> mass (in-place update)

    Behavior:
        - If the sum is 0, leave the distribution unchanged.
        - Otherwise, divide each mass by the total.
    """
    raise NotImplementedError


# --------------------------
# Inference by enumeration
# --------------------------

def all_assignments(net: BayesNet, fixed: dict):
    """Generate all complete assignments consistent with 'fixed' (dict var->value)."""
    vars_free = [v for v in net.order if v not in fixed]
    doms = [net.values(v) for v in vars_free]
    for combo in itertools.product(*doms):
        asg = dict(fixed)
        asg.update({v: val for (v, val) in zip(vars_free, combo)})
        yield asg


def query_distribution(net: BayesNet, qvar: str, evidence: dict):
    """Build P(qvar | evidence) by summing joint probabilities over complete assignments."""
    dist = {val: 0.0 for val in net.values(qvar)}
    for val in net.values(qvar):
        fixed = dict(evidence)
        fixed[qvar] = val
        total = 0.0
        for asg in all_assignments(net, fixed):
            total += joint_probability(net, asg)
        update(dist, val, total)
    normalize(dist)
    return dist


def parse_query(line: str):
    """Parse lines like:
        'Burglary'
        'JohnCalls | Alarm = T'
        'Burglary | JohnCalls = T, MaryCalls = T'
    Returns (qvar, evidence_dict).
    """
    segs = [s.strip() for s in line.split("|")]
    qvar = segs[0].strip()
    evidence = {}
    if len(segs) > 1:
        rhs = segs[1]
        if rhs:
            parts = [p.strip() for p in rhs.split(",")]
            for p in parts:
                if not p:
                    continue
                lhs, val = [t.strip() for t in p.split("=")]
                evidence[lhs] = val
    return qvar, evidence


def print_distribution(qvar, dist, order):
    """Print like:  P(T) = 0.001, P(F) = 0.999   (3 decimals)."""
    parts = [f"P({v}) = {dist[v]:.3f}" for v in order]
    # print(qvar)
    print(", ".join(parts))
    print()


def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python bn_tunner_solution.py <network.txt>")
    path = sys.argv[1]
    print(f'Loading file "{path}".')
    print('Type \'quit\' to stop the simulation.')

    net = parse_bn(path)

    try:
        while True:
            line = input().strip()
            if not line:
                continue
            if line.lower() == "quit":
                break

            qvar, evidence = parse_query(line)
            if qvar not in net.domains:
                print(f"Unknown variable '{qvar}'.")
                continue

            # Basic validation on evidence
            ok = True
            for ev, val in evidence.items():
                if ev not in net.domains:
                    print(f"Unknown evidence variable '{ev}'.")
                    ok = False
                    break
                if val not in net.domains[ev]:
                    print(f"Value '{val}' not in domain of '{ev}'.")
                    ok = False
                    break
            if not ok:
                continue

            dist = query_distribution(net, qvar, evidence)
            ev_str = ", ".join([f"{k} = {v}" for k, v in evidence.items()])
            header = qvar + (" | " + ev_str if ev_str else "")
            print_distribution(header, dist, net.values(qvar))
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
