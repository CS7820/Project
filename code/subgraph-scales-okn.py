import os 
from rdflib import URIRef, Graph, Namespace, Literal
from rdflib import OWL, RDF, RDFS, XSD, TIME
from rdflib.plugins.parsers.notation3 import ParserError

input_path = "../scales-okn/scales_cross_sectional"

pfs = {
"scales": Namespace("http://schemas.scales-okn.org/rdf/scales#"),
"rdf": RDF,
"rdfs": RDFS,
"xsd": XSD,
"owl": OWL,
}

def init_kg():
    kg = Graph()
    for prefix in pfs:
        kg.bind(prefix, pfs[prefix])
    return kg

def parse_subgraph():
    '''
    Attempts to parse through generated TTL files to represent Cases, Courts, Charges, and Agents along with their roles for a case.

    Uses rdflib Graph to generate s,p,o per TTL file
    '''
    ## TODO: Automatically handle directories/file selection on TTL
    subgraph_types = ["Cases", "Courts", "Charges", "Agents"]
    
    # Loop over TTL files for the current subgraph type
    scales_root_files =  os.listdir(input_path)

    # Try to parse the graph
    graph = init_kg()
    subgraph = init_kg()
    for file in scales_root_files:
        if os.path.isfile(os.path.join(input_path, file)):
            graph.parse(os.path.join(input_path, file))
        else:  # directory
            scales_district_dir = os.path.join(input_path, file)
            list_at_scales_dist_dir =  os.listdir(scales_district_dir)

            for year in list_at_scales_dist_dir:
                district_year_dir = os.path.join(scales_district_dir, year)
                ttl_files = os.listdir(district_year_dir)
                for ttl_file in ttl_files:
                    curr_ttl = os.path.join(district_year_dir, ttl_file)
                    with open(curr_ttl, "r") as inp:
                        graph.parse(inp)
    
    for case_subject, _, _ in graph.triples((None, RDF.type, pfs["scales"]["Case"])):
        # graph.value(case_subject, RDFS.label)
        # graph.value(case_subject, pfs["ontology"]["description"])
        pass
    for court_subject, _, _ in graph.triples((None, RDF.type, pfs["scales"]["Court"])):
        # Information about the courts 
        pass
    for charge_subject, _, _ in graph.triples((None, RDF.type, pfs["scales"]["Charge"])):
        # Information about the charges 
        pass
    for agent_subject, _, _ in graph.triples((None, RDF.type, pfs["scales"]["Agent"])):
        # graph.value(agent_subject, FOAF.name)
        # graph.value(agent_subject, pfs["ontology"]["role"])
        # graph.value(agent_subject, pfs["ontology"]["associatedWith"])
        pass
    

if __name__ == "__main__":
    parse_subgraph()