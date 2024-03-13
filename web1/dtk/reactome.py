
import logging
from dtk.log_setup import setupLogging
logger = logging.getLogger(__name__)

ROOT_ID = '[root]'

class Prot:
    def __init__(self, data, db):
        self.id = data['identifier']
        geneNames = data['geneName']
        if geneNames:
            geneName = geneNames[0]
            self.name = f'{geneName} ({self.id})'
        else:
            self.name = data['displayName']
        self.hasDiagram = False

    def __repr__(self):
        return f'({self.name} [{self.id}])'



class Pathway:
    def __init__(self, data, db):
        self.db = db
        #print("init with ", data)
        self.name = data["displayName"]
        self.id = data["stId"]
        self.hasDiagram = data.get('hasDiagram', False)
        if not hasattr(data, 'labels') or 'Pathway' in data.labels:
            self.type = 'pathway'
        else:
            self.type = 'event'

    def __repr__(self):
        return f'({self.name} [{self.id}])'

    def get_neighbors(self):
        if self.id == ROOT_ID:
            return []

        results = self.db.run("""
            MATCH (p:Event)-[:precedingEvent|:followingEvent]-(pp:Event{stId:$stId})
            RETURN p
            """, {
                "stId": self.id,
            })
        return [Pathway(result[0], db=self.db) for result in results]


    def get_sub_pathways(self):
        if self.id == ROOT_ID:
            return Reactome.get_toplevel_pathways(self.db)
        results = self.db.run("""
            MATCH (p:Event)<-[:hasEvent]-(pp:Event{stId:$stId})
            RETURN p
            """, {
                "stId": self.id,
            })
        return [Pathway(result[0], db=self.db) for result in results]

    def get_proteins(self):
        """Returns the set of proteins that are directly a part of this event.

        If you want all proteins of all subevents/subpathways, use
        get_all_sub_proteins.
        """
        results = self.db.run("""
            MATCH (r:ReactionLikeEvent{stId:$stId})-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity)-[:referenceEntity]->(re:ReferenceEntity)
            RETURN DISTINCT re
            """, {
                "stId": self.id
            })
        out = [Prot(result[0], db=self.db) for result in results]
        out.sort(key=lambda x:x.name)
        return out

    def get_all_sub_proteins(self):
        """Returns all proteins part of this pathway or its subpathways/events."""
        if self.type == 'pathway':
            results = self.db.run("""
                MATCH (p:Pathway{stId:$stId})-[:hasEvent*]->(r:ReactionLikeEvent)-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity)-[:referenceEntity]->(re:ReferenceEntity)
                RETURN DISTINCT re
                """, {
                    "stId": self.id
                })
            return [Prot(result[0], db=self.db) for result in results]
        else:
            results = self.db.run("""
                MATCH (r:ReactionLikeEvent{stId:$stId})-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity)-[:referenceEntity]->(re:ReferenceEntity)
                RETURN DISTINCT re
                """, {
                    "stId": self.id
                })
            return [Prot(result[0], db=self.db) for result in results]


class ProtSet:
    def __init__(self, pw, children):
        self.pw = pw
        self.children = children

    def all_protsets(self):
        out = [self]
        for child in self.children:
            out.extend(child.all_protsets())
        return out

    def __iter__(self):
        seen = set()
        for child in self.children:
            for prot in child:
                if prot not in seen:
                    seen.add(prot)
                    yield prot

class SingleProtSet:
    def __init__(self, prot):
        self.prot = prot
    def __iter__(self):
        yield self.prot
    def all_protsets(self):
        return []

def generate_protset(r, pw, reaction2prots):
    if pw.type == 'pathway':
        children = [generate_protset(r, subpw, reaction2prots)
                    for subpw in pw.get_sub_pathways()]
        return ProtSet(pw, children)
    else:
        children = [SingleProtSet(prot)
                    for prot in reaction2prots.get(pw.id, [])]
        return ProtSet(pw, children)


def reactome_host():
    import os
    if not os.path.exists("/.dockerenv"):
        # We're not inside a container, so reactome should be running on localhost.
        return 'localhost'
    else:
        # We're inside a container, reactome should be a sibling container available from
        # the host's internal IP.
        #
        # Seems there isn't a great non-hacky way to do this, but this isn't too
        # bad.  Weirdly windows and mac have a nice solution, but linux does not.
        #
        # We're relying on being inside AWS and having access to the instance metadata,
        # which knows what our local ip is.
        #
        # There are some other ideas here:
        # https://stackoverflow.com/questions/22944631/how-to-get-the-ip-address-of-the-docker-host-from-inside-a-docker-container
        import requests
        return requests.get('http://169.254.169.254/latest/meta-data/local-ipv4').text



class Reactome:
    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(f"bolt://{reactome_host()}:7687")
        self.db = self.driver.session()

    @classmethod
    def get_toplevel_pathways(cls, db):
        results = db.run("""
            MATCH (p:Pathway{speciesName:"Homo sapiens"})
            WHERE NOT (:Pathway)-[:hasEvent]->(p)
            RETURN p
            """)
        return [Pathway(result[0], db=db) for result in results]

    def get_old_to_new_ids(self):
        results = self.db.run("""
            MATCH (p:Pathway{speciesName:"Homo sapiens"})
            RETURN p.oldStId, p.stId
            """)
        return {x[0]:x[1] for x in results}


    def get_pathway(self, stId):
        if stId == ROOT_ID:
            return Pathway({'stId': ROOT_ID,  'displayName': 'Root', 'normalPathway': False}, db=self.db)
        results = self.db.run("""
            MATCH (p:Event{stId:$stId})
            RETURN p
            """, {
                "stId": stId
            })
        return Pathway(next(iter(results))[0], db=self.db)

    def get_pathway_id_name_map(self):
        results = self.db.run("""
            MATCH (p:Event{speciesName:"Homo sapiens"})
            RETURN p.stId, p.displayName
            """)
        return {x[0]:x[1] for x in results}

    def get_pathway_hierarchy(self):
        results = self.db.run("""
            MATCH (p:Event{speciesName:"Homo sapiens"})<-[:hasEvent]-(pp:Event{speciesName:"Homo sapiens"})
            RETURN pp.stId, p.stId
            """
            )

        pairs = [(str(result[0]), str(result[1])) for result in results]

        toplevel = self.get_toplevel_pathways(self.db)
        toplevel_pairs = [(ROOT_ID, p.id) for p in toplevel]
        pairs.extend(toplevel_pairs)

        from dtk.data import MultiMap
        return MultiMap(pairs)

    def get_all_pathways(self):
        results = self.db.run("""
            MATCH (p:Event{speciesName:"Homo sapiens"})
            RETURN p
            """)
        pws = [Pathway(result[0], db=self.db) for result in results]
        return pws

    def get_pathways(self, stIds):
        results = self.db.run("""
            MATCH (p:Event)
            WHERE p.stId IN {stIds}
            RETURN p
            """, {
                "stIds": stIds
            })
        pws = [Pathway(result[0], db=self.db) for result in results]
        id2pw = {pw.id:pw for pw in pws}
        return [id2pw[stId] for stId in stIds]


    def get_pathways_with_prot(self, prot):
        # Find the reactions.
        results = self.db.run("""
            MATCH (r:ReactionLikeEvent)-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity)-[:referenceEntity]->(:ReferenceEntity{identifier:$uniprot})
            RETURN DISTINCT r
            """, {
                "uniprot": prot
            })
        out = [Pathway(result[0], db=self.db) for result in results]

        # Find the pathways containing that reaction (& above).
        results = self.db.run("""
            MATCH (r:ReactionLikeEvent)-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity)-[:referenceEntity]->(:ReferenceEntity{identifier:$uniprot}),
                (p:Event)-[:hasEvent*]->(r)
            RETURN DISTINCT p
            """, {
                "uniprot": prot
            })
        out += [Pathway(result[0], db=self.db) for result in results]
        return out

    def get_reactions_to_prots(self):
        """Finds all human proteins in the reactions.

        Note that this will exclude e.g. flu proteins.
        """

        # Find the reactions.
        results = self.db.run("""
            MATCH (r:ReactionLikeEvent{speciesName:"Homo sapiens"})-[:input|output|catalystActivity|physicalEntity|regulatedBy|regulator|hasComponent|hasMember|hasCandidate*]->(pe:PhysicalEntity{speciesName:"Homo sapiens"})-[:referenceEntity]->(re:ReferenceEntity)
            RETURN DISTINCT r.stId, re.identifier
            """
            )
        from dtk.data import MultiMap
        logger.info("Generating results")
        mm = MultiMap(results)
        logger.info("Got %d results", len(mm.fwd_map()))
        return mm

def score_pathways(pathways, score_list):
    output = []
    for pathway in pathways:
        gl_prots = pathway.get_all_sub_proteins()
        # TODO: Does this include non-proteins?  I think yes.
        gl = [x.id for x in gl_prots]
        from scripts.glf import run_single_glf
        glf_out = run_single_glf((pathway.id, gl), score_list, 'wOR')
        output.append(glf_out)
    return output



if __name__ == "__main__":
    # You can run this as a basic sanity check that you can connect with neo4j.
    # XXX As of 2022-03-04, the setupLogging() call dies deep in django,
    # XXX because an attempt to import html pulls in dtk.html rather than
    # XXX the python package. Possibly this used to reside in some directory
    # XXX other than dtk?
    setupLogging()
    r = Reactome()
    reactions_to_prots = r.get_reactions_to_prots().fwd_map()


    top_pws = Reactome.get_toplevel_pathways(r.db)
    for pw in top_pws:
        logger.info("Making protset for %s", pw)
        ps = generate_protset(r, pw, reactions_to_prots)

    pw = r.get_pathways_with_prot('P29275')
    assert len(pw) > 0, "There should be pathways with this protein"
    print("Successfully found pathways", pw)
