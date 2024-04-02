import os 
import pandas as pd
from rdflib import URIRef, Graph, Namespace, Literal
from rdflib import OWL, RDF, RDFS, XSD, TIME
from rdflib.plugins.parsers.notation3 import ParserError

# File Parameters
data_path = "../dataset/cook-county" 
input_file = "Sentencing.csv"

ttl_output_path = "../dataset/cook-county/ttl-files"
txt_output_path = "../dataset/cook-county/txt-format-files"
write_file = "cook-county-cases-guilty-verdict.txt"
write_path = os.path.join(txt_output_path, write_file)
if(os.path.exists(write_path)):
    outFile = open(write_path, "a")
else:
    outFile = open(write_path, "w")


# Columns to Consider in Materialization
csv_columns = [
"CASE_ID", "OFFENSE_CATEGORY", "PRIMARY_CHARGE_FLAG",
"CHARGE_ID", "CHARGE_VERSION_ID", "DISPOSITION_CHARGED_OFFENSE_TITLE",
"CHARGE_COUNT", "DISPOSITION_CHARGED_CHAPTER", "DISPOSITION_CHARGED_ACT",
"DISPOSITION_CHARGED_SECTION", "DISPOSITION_CHARGED_CLASS", "DISPOSITION_CHARGED_AOIC",
"CHARGE_DISPOSITION", "CHARGE_DISPOSITION_REASON", "SENTENCE_JUDGE", 
"SENTENCE_COURT_NAME", "SENTENCE_COURT_FACILITY", "SENTENCE_PHASE",
"SENTENCE_TYPE", "CURRENT_SENTENCE_FLAG", "COMMITMENT_TYPE",
"COMMITMENT_TERM", "COMMITMENT_UNIT", "AGE_AT_INCIDENT",
"LAW_ENFORCEMENT_AGENCY", "FELONY_REVIEW_RESULT", "UPDATED_OFFENSE_CATEGORY"
]

sub_graph_columns = [
# General Case Info
"CASE_ID", "OFFENSE_CATEGORY", "UPDATED_OFFENSE_CATEGORY"
# Court Info
"SENTENCE_COURT_NAME",
# Agent - Judge
"SENTENCE_JUDGE"
# Charge/Sentence Detail
"SENTENCE_TYPE",
"CHARGE_ID", "CHARGE_VERSION_ID", "DISPOSITION_CHARGED_OFFENSE_TITLE",
"CHARGE_COUNT"
# Committment as Quantity Info
"COMMITMENT_TYPE", "COMMITMENT_TERM", "COMMITMENT_UNIT"

]

name_space = "https://kastle-lab.org/lod"


# Prefixes of used namespaces
pfs = {
"sdg-onto": Namespace(f"{name_space}/ontology/"), # Ontology Piece
"sdg-res": Namespace(f"{name_space}/resource/"),  # Instanciated Pieces for Ontology
"rdf": RDF,
"rdfs": RDFS,
"xsd": XSD,
"owl": OWL,
}

# Type Shortcut
isA = pfs["rdf"]["type"]

def init_kg():
    kg = Graph()
    for prefix in pfs:
        kg.bind(prefix, pfs[prefix])
    return kg

def generate_entityDict(type, col):
    count = 0
    entityDict = dict()
    for label in col:
        if label not in entityDict:
            entityDict[label] = f"{type}{count}"
            count+=1
    return entityDict
          
def bind_spo(subject, property, object):
    graph.add( (subject, property, object) )

def write_spo(s,p,o):
    outFile.write(f"{s}\t{p}\t{o}\n")

def materialize_graph(graph, begin, end, output_file):
    csv_path = os.path.join(data_path, input_file)
    df = pd.read_csv(csv_path)
    
    caseDict = generate_entityDict("Case", df['CASE_ID'])
    judgeDict = generate_entityDict("Judge", df['SENTENCE_JUDGE'])
    courtDict = generate_entityDict("Court", df['SENTENCE_COURT_NAME'])
    offenseCatDict = generate_entityDict("OffenseCategory", df['UPDATED_OFFENSE_CATEGORY'])
    chargeCOTDict = generate_entityDict("ChargedOffenseTitle", df['DISPOSITION_CHARGED_OFFENSE_TITLE'])
    commTypeDict = generate_entityDict("CommitmentType", df['COMMITMENT_TYPE'])
    sentenceTypeDict = generate_entityDict("SentenceType", df['SENTENCE_TYPE'])
    # Role URIs
    judge_role_uri = pfs['sdg-onto']["AssignedJudge"]

    previous = 0
    for df_ind, row in df.iterrows():
        if df_ind < begin:
            continue
        if df_ind == end:
            break
        #  Materialize Case
        ## General Case Info
        ## "CASE_ID"

        caseID = caseDict[row['CASE_ID']]
        case_uri = pfs['sdg-res'][caseID]
        graph.add( (case_uri, isA, pfs['sdg-onto']['Case']) )

        write_spo(caseID, 'type', 'Case')
        #  Materialize OffenseCategory
        ## "OFFENSE_CATEGORY", "UPDATED_OFFENSE_CATEGORY"
        offenseID = offenseCatDict[row['UPDATED_OFFENSE_CATEGORY']]
        offense_uri = pfs['sdg-res'][offenseID]
        graph.add( (offense_uri, isA, pfs['sdg-onto']['OffenseCategory']) )
        bind_spo(offense_uri, pfs['sdg-onto']['offenseAsString'], Literal(row['UPDATED_OFFENSE_CATEGORY'], datatype=XSD.string))
        bind_spo(case_uri, pfs["sdg-onto"]['hasOffenseCategory'], offense_uri)

        write_spo(offenseID, 'type', 'OffenseCategory')
        write_spo(offenseID, 'offenseAsString', row['UPDATED_OFFENSE_CATEGORY'])
        write_spo(caseID, 'hasOffenseCategory', offenseID)

        #  Materialize Court
        ## Court Info
        ## "SENTENCE_COURT_NAME",
        courtID = courtDict[row['SENTENCE_COURT_NAME']]

        court_names = list(courtDict.keys())
        court_ids = list(courtDict.values())
        index = court_ids.index(courtID)

        court_name = court_names[index]
        court_uri = pfs['sdg-res'][courtID]
        graph.add( (court_uri, isA, pfs["sdg-onto"]['Court']) )
        bind_spo(court_uri, pfs['sdg-onto']['hasName'], Literal(court_name, datatype=XSD.string))
        bind_spo(case_uri, pfs['sdg-onto']['isInCourt'], court_uri)

        write_spo(courtID, 'type', 'Court')
        write_spo(courtID, 'hasName', court_name)
        write_spo(caseID, 'isInCourt', courtID)
        
        #  Materialize Judge
        ## Agent - Judge
        ## "SENTENCE_JUDGE"
        judge_name = row['SENTENCE_JUDGE']
        judgeID = judgeDict[judge_name]
        judge_uri = pfs['sdg-res'][judgeID]        
        graph.add( (judge_uri, isA, pfs['sdg-onto']['Judge']) )
        bind_spo(judge_uri, pfs['sdg-onto']['performsRole'], judge_role_uri)
        bind_spo(case_uri, pfs['sdg-onto']['hasAgent'], judge_uri)

        write_spo(judgeID, 'type', 'Judge')
        write_spo(judgeID, 'performsRole', 'AssignedJudge')
        write_spo(caseID, 'hasAgent', judgeID)

        '''
            One case can have many sentencing charges.
            `previous` controls whether a new case has been identified
            to reduce on repeated materialization
        '''
        if(previous != caseID):
            previous = caseID
            # Filter original dataframe to sentencing information on current case
            case_sentences_df = df[df['CASE_ID'] == row['CASE_ID']]
            sentence_charges_df = case_sentences_df[['SENTENCE_TYPE',
                                                     'CHARGE_ID', 'CHARGE_VERSION_ID', 
                                                     'DISPOSITION_CHARGED_OFFENSE_TITLE','CHARGE_COUNT', 'PRIMARY_CHARGE_FLAG',
                                                     'COMMITMENT_TYPE', 'COMMITMENT_TERM', 'COMMITMENT_UNIT']]
            
            for ind, charge_row in sentence_charges_df.iterrows(): # For each charge, materialize a new sentencing/charge entity
                # Materializing Sentencing
                ## Charge/Sentence Detail
                ## "SENTENCE_TYPE",
                sentence_uri = pfs['sdg-onto'][f"Case{caseID}.Sentence{ind}"]
                sent_type_label = f"{sentenceTypeDict[charge_row['SENTENCE_TYPE']]}"
                sent_type_uri = pfs['sdg-onto'][sent_type_label]
                graph.add( (sentence_uri, isA, pfs['sdg-onto']['Sentence']) )
                graph.add( (sent_type_uri, isA, pfs['sdg-onto']['SentenceType']) )
                
                bind_spo(case_uri, pfs['sdg-onto']['hasSentence'], sentence_uri)
                bind_spo(sentence_uri, pfs['sdg-onto']['hasSentenceType'], sent_type_uri)  # Do cases have many sentences?
                bind_spo(sent_type_uri, pfs['sdg-onto']['typeAsString'], Literal(charge_row['SENTENCE_TYPE'], datatype=XSD.string))
                
                write_spo(f"Case{caseID}.Sentence{ind}", 'type', 'Sentence')
                write_spo(sent_type_label, "type", "SentenceType")
                write_spo(caseID, "hasSentence", f"Case{caseID}.Sentence{ind}")
                write_spo(f"Case{caseID}.Sentence{ind}", "hasSentenceType", sent_type_label)
                write_spo(sent_type_label, "typeAsString", charge_row['SENTENCE_TYPE'])

                # Materialize Charge
                ## "CHARGE_ID", "CHARGE_VERSION_ID", "DISPOSITION_CHARGED_OFFENSE_TITLE",
                ## "CHARGE_COUNT"
                curr_charge_id = charge_row['CHARGE_ID']
                curr_cot = charge_row['DISPOSITION_CHARGED_OFFENSE_TITLE']
                curr_primary_charge_flag = charge_row['PRIMARY_CHARGE_FLAG']
                cot_id = chargeCOTDict[curr_cot]
                charge_label = f"Case{caseID}.Charge{ind}.{curr_charge_id}"
                charge_uri = pfs['sdg-onto'][charge_label]
                charge_cot_uri = pfs['sdg-onto'][f"{cot_id}"]
                graph.add( (charge_uri, isA, pfs['sdg-onto']['Charge']) )
                graph.add( (charge_cot_uri, isA, pfs['sdg-onto']['ChargedOffenseTitle']) )

                bind_spo(charge_uri, pfs['sdg-onto']['hasChargedOffenseTitle'],charge_cot_uri)
                bind_spo(charge_uri, pfs['sdg-onto']['isPrimaryCharge'],Literal(curr_primary_charge_flag, datatype=XSD.boolean))
                bind_spo(charge_cot_uri, pfs['sdg-onto']['titleAsString'], Literal(charge_row['DISPOSITION_CHARGED_OFFENSE_TITLE'], datatype=XSD.string))
                ### Bind Sentence to Charge
                bind_spo(sentence_uri, pfs['sdg-onto']['hasCharge'], charge_uri)

                write_spo(charge_label, "type", "Charge")
                write_spo(curr_cot, "type", "ChargedOffenseTitle")
                write_spo(charge_label, "hasChargedOffenseTitle", curr_cot)
                write_spo(charge_label, "isPrimaryCharge", curr_primary_charge_flag)
                write_spo(curr_cot, "titleAsString", charge_row['DISPOSITION_CHARGED_OFFENSE_TITLE'])
                write_spo(f"Case{caseID}.Sentence{ind}", "hasCharge", charge_label)
                
                # Materialize Commitment as a Quanity
                ## Committment as Quantity Info
                ## "COMMITMENT_TYPE", "COMMITMENT_TERM", "COMMITMENT_UNIT"
                commType = commTypeDict[row['COMMITMENT_TYPE']]
                commNumb = charge_row['COMMITMENT_TERM']
                commUnit = charge_row['COMMITMENT_UNIT']
                
                commit_label = f"Case{caseID}.Charge{ind}.Commitment"
                comm_uri = pfs['sdg-res'][f"{commit_label}"]
                commType_uri = pfs['sdg-res'][commType]
                graph.add((comm_uri, isA, pfs["sdg-onto"]["Commitment"]))
                graph.add((commType_uri, isA, pfs["sdg-onto"]["CommitmentType"]))
                
                bind_spo(comm_uri, pfs['sdg-onto']['hasCommitmentType'], commType_uri)
                bind_spo(commType_uri, pfs['sdg-onto']['typeAsString'], Literal(charge_row['COMMITMENT_TYPE'], datatype=XSD.string))
                
                write_spo(commit_label, "type", "Commitment")
                write_spo(commType, "type", "CommitmentType")
                write_spo(commit_label, "hasCommitmentType", commType)
                write_spo(commit_label, "typeAsString", charge_row['COMMITMENT_TYPE'])

                # Only bind Value if the term and unit are not blank
                if(commNumb != "" and commUnit != ""):
                    commValue_uri = pfs['sdg-res'][f"{commit_label}.Value"]
                    commUnit_uri = pfs['sdg-res'][f"{commit_label}.Value.Unit"]
                    graph.add((commValue_uri, isA, pfs["sdg-onto"]["CommitmentValue"]))
                    graph.add((commUnit_uri, isA, pfs["sdg-onto"]["CommitmentUnit"]))
                    
                    bind_spo(comm_uri, pfs['sdg-onto']['hasCommitmentValue'], commValue_uri)
                    bind_spo(commValue_uri, pfs['sdg-onto']['hasNumericValue'], Literal(commNumb, datatype=XSD.double))
                    bind_spo(commValue_uri, pfs['sdg-onto']['hasUnit'], commUnit_uri)

                    write_spo(f"{commit_label}.Value", "type", "CommitmentValue")
                    write_spo(f"{commit_label}.Value.Unit", "type", "CommitmentUnit")
                    write_spo(commit_label, "hasCommitmentValue", f"{commit_label}.Value")
                    write_spo(f"{commit_label}.Value", "hasNumericValue", commNumb)
                    write_spo(f"{commit_label}.Value", "hasUnit", commUnit)
                ### Charge-to-Commitment
                bind_spo(charge_uri, pfs['sdg-onto']['hasCommitment'], comm_uri)
                
                write_spo(charge_label, 'hasCommitment', commit_label)

    temp = graph.serialize(format="turtle", encoding="utf-8", 
                           destination=os.path.join(ttl_output_path, output_file))        

def cleanup_duplicates():
    '''
        Due to the nature of singular cases containing multiple charges, 
        there is the possibility duplicate facts are added to the KG.

        This is automatically cleaned from the TTL files from the Triplestore.
        cleanup_duplicates will clean up the .txt file, or KG written to a text file.
    '''
    with open(write_path, "r") as inp:
        lines = [ line for line in inp.readlines()]
    unique_set = set()
    for line in lines:
        unique_set.add(line)

    with open(write_path, "w") as out:
        for u in unique_set:
            out.write(u)

if __name__ == "__main__":
    count = 1
    step = 10000
    begin = 0
    end = step
    for i in range(0, 300000, step):
        output_file = ""
        if count < 10:
            output_file = f"cook-county-cases-guilty-verdict-0{count}.ttl"
        else:
            output_file = f"cook-county-cases-guilty-verdict-{count}.ttl"

        graph = init_kg() # Reset graph per partial-complete
        materialize_graph(graph, begin, end, output_file)
        print(f"Materialization partial complete: {output_file}")
        begin = begin+step
        end = end+step
        count +=1
    outFile.close()
    cleanup_duplicates()
