from peewee import *

database = MySQLDatabase('chembl_20', **{'user': 'root'})

class UnknownField(object):
    pass

class BaseModel(Model):
    class Meta:
        database = database

class ActionType(BaseModel):
    action_type = CharField(primary_key=True)
    description = CharField()
    parent_type = CharField(null=True)

    class Meta:
        db_table = 'action_type'

class RelationshipType(BaseModel):
    relationship_desc = CharField(null=True)
    relationship_type = CharField(primary_key=True)

    class Meta:
        db_table = 'relationship_type'

class AssayType(BaseModel):
    assay_desc = CharField(null=True)
    assay_type = CharField(primary_key=True)

    class Meta:
        db_table = 'assay_type'

class ChemblIdLookup(BaseModel):
    chembl = CharField(db_column='chembl_id', primary_key=True)
    entity = IntegerField(db_column='entity_id')
    entity_type = CharField()
    status = CharField()

    class Meta:
        db_table = 'chembl_id_lookup'

class CellDictionary(BaseModel):
    cell_description = CharField(null=True)
    cell = PrimaryKeyField(db_column='cell_id')
    cell_name = CharField()
    cell_source_organism = CharField(null=True)
    cell_source_tax = IntegerField(db_column='cell_source_tax_id', null=True)
    cell_source_tissue = CharField(null=True)
    cellosaurus = CharField(db_column='cellosaurus_id', null=True)
    chembl = ForeignKeyField(db_column='chembl_id', null=True, rel_model=ChemblIdLookup, to_field='chembl')
    cl_lincs = CharField(db_column='cl_lincs_id', null=True)
    clo = CharField(db_column='clo_id', null=True)
    efo = CharField(db_column='efo_id', null=True)

    class Meta:
        db_table = 'cell_dictionary'

class ConfidenceScoreLookup(BaseModel):
    confidence_score = PrimaryKeyField()
    description = CharField()
    target_mapping = CharField()

    class Meta:
        db_table = 'confidence_score_lookup'

class CurationLookup(BaseModel):
    curated_by = CharField(primary_key=True)
    description = CharField()

    class Meta:
        db_table = 'curation_lookup'

class Docs(BaseModel):
    abstract = TextField(null=True)
    authors = CharField(null=True)
    chembl = ForeignKeyField(db_column='chembl_id', rel_model=ChemblIdLookup, to_field='chembl', unique=True)
    doc = PrimaryKeyField(db_column='doc_id')
    doc_type = CharField()
    doi = CharField(null=True)
    first_page = CharField(null=True)
    issue = CharField(index=True, null=True)
    journal = CharField(index=True, null=True)
    last_page = CharField(null=True)
    pubmed = IntegerField(db_column='pubmed_id', null=True, unique=True)
    title = CharField(null=True)
    volume = CharField(index=True, null=True)
    year = IntegerField(index=True, null=True)

    class Meta:
        db_table = 'docs'

class Source(BaseModel):
    src_description = CharField(null=True)
    src = PrimaryKeyField(db_column='src_id')
    src_short_name = CharField(null=True)

    class Meta:
        db_table = 'source'

class TargetType(BaseModel):
    parent_type = CharField(null=True)
    target_desc = CharField(null=True)
    target_type = CharField(primary_key=True)

    class Meta:
        db_table = 'target_type'

class TargetDictionary(BaseModel):
    chembl = ForeignKeyField(db_column='chembl_id', rel_model=ChemblIdLookup, to_field='chembl')
    organism = CharField(index=True, null=True)
    pref_name = CharField(index=True)
    species_group_flag = IntegerField()
    target_type = ForeignKeyField(db_column='target_type', null=True, rel_model=TargetType, to_field='target_type')
    tax = IntegerField(db_column='tax_id', index=True, null=True)
    tid = PrimaryKeyField()

    class Meta:
        db_table = 'target_dictionary'

class Assays(BaseModel):
    assay_category = CharField(null=True)
    assay_cell_type = CharField(null=True)
    assay = PrimaryKeyField(db_column='assay_id')
    assay_organism = CharField(null=True)
    assay_strain = CharField(null=True)
    assay_subcellular_fraction = CharField(null=True)
    assay_tax = IntegerField(db_column='assay_tax_id', null=True)
    assay_test_type = CharField(null=True)
    assay_tissue = CharField(null=True)
    assay_type = ForeignKeyField(db_column='assay_type', null=True, rel_model=AssayType, to_field='assay_type')
    bao_format = CharField(index=True, null=True)
    cell = ForeignKeyField(db_column='cell_id', null=True, rel_model=CellDictionary, to_field='cell')
    chembl = ForeignKeyField(db_column='chembl_id', rel_model=ChemblIdLookup, to_field='chembl', unique=True)
    confidence_score = ForeignKeyField(db_column='confidence_score', null=True, rel_model=ConfidenceScoreLookup, to_field='confidence_score')
    curated_by = ForeignKeyField(db_column='curated_by', null=True, rel_model=CurationLookup, to_field='curated_by')
    description = CharField(null=True)
    doc = ForeignKeyField(db_column='doc_id', rel_model=Docs, to_field='doc')
    relationship_type = ForeignKeyField(db_column='relationship_type', null=True, rel_model=RelationshipType, to_field='relationship_type')
    src_assay = CharField(db_column='src_assay_id', null=True)
    src = ForeignKeyField(db_column='src_id', rel_model=Source, to_field='src')
    tid = ForeignKeyField(db_column='tid', null=True, rel_model=TargetDictionary, to_field='tid')

    class Meta:
        db_table = 'assays'

class DataValidityLookup(BaseModel):
    data_validity_comment = CharField(primary_key=True)
    description = CharField(null=True)

    class Meta:
        db_table = 'data_validity_lookup'

class MoleculeDictionary(BaseModel):
    availability_type = IntegerField(null=True)
    black_box_warning = IntegerField()
    chebi_par = IntegerField(db_column='chebi_par_id', null=True)
    chembl = ForeignKeyField(db_column='chembl_id', rel_model=ChemblIdLookup, to_field='chembl', unique=True)
    chirality = IntegerField()
    dosed_ingredient = IntegerField()
    first_approval = IntegerField(null=True)
    first_in_class = IntegerField()
    indication_class = CharField(null=True)
    inorganic_flag = IntegerField()
    max_phase = IntegerField(index=True)
    molecule_type = CharField(null=True)
    molregno = PrimaryKeyField()
    natural_product = IntegerField()
    oral = IntegerField()
    parenteral = IntegerField()
    polymer_flag = IntegerField(null=True)
    pref_name = CharField(index=True, null=True)
    prodrug = IntegerField()
    structure_type = CharField()
    therapeutic_flag = IntegerField(index=True)
    topical = IntegerField()
    usan_stem = CharField(null=True)
    usan_stem_definition = CharField(null=True)
    usan_substem = CharField(null=True)
    usan_year = IntegerField(null=True)

    class Meta:
        db_table = 'molecule_dictionary'

class CompoundRecords(BaseModel):
    compound_key = CharField(index=True, null=True)
    compound_name = CharField(null=True)
    doc = ForeignKeyField(db_column='doc_id', rel_model=Docs, to_field='doc')
    molregno = ForeignKeyField(db_column='molregno', null=True, rel_model=MoleculeDictionary, to_field='molregno')
    record = PrimaryKeyField(db_column='record_id')
    src_compound = CharField(db_column='src_compound_id', index=True, null=True)
    src = ForeignKeyField(db_column='src_id', rel_model=Source, to_field='src')

    class Meta:
        db_table = 'compound_records'

class Activities(BaseModel):
    activity_comment = CharField(null=True)
    activity = PrimaryKeyField(db_column='activity_id')
    assay = ForeignKeyField(db_column='assay_id', rel_model=Assays, to_field='assay')
    bao_endpoint = CharField(null=True)
    data_validity_comment = ForeignKeyField(db_column='data_validity_comment', null=True, rel_model=DataValidityLookup, to_field='data_validity_comment')
    doc = ForeignKeyField(db_column='doc_id', null=True, rel_model=Docs, to_field='doc')
    molregno = ForeignKeyField(db_column='molregno', null=True, rel_model=MoleculeDictionary, to_field='molregno')
    pchembl_value = DecimalField(index=True, null=True)
    potential_duplicate = IntegerField(null=True)
    published_relation = CharField(index=True, null=True)
    published_type = CharField(index=True, null=True)
    published_units = CharField(index=True, null=True)
    published_value = DecimalField(index=True, null=True)
    qudt_units = CharField(null=True)
    record = ForeignKeyField(db_column='record_id', rel_model=CompoundRecords, to_field='record')
    standard_flag = IntegerField(null=True)
    standard_relation = CharField(index=True, null=True)
    standard_type = CharField(index=True, null=True)
    standard_units = CharField(index=True, null=True)
    standard_value = DecimalField(index=True, null=True)
    uo_units = CharField(null=True)

    class Meta:
        db_table = 'activities'

class ActivityStdsLookup(BaseModel):
    definition = CharField(null=True)
    normal_range_max = DecimalField(null=True)
    normal_range_min = DecimalField(null=True)
    standard_type = CharField()
    standard_units = CharField()
    std_act = PrimaryKeyField(db_column='std_act_id')

    class Meta:
        db_table = 'activity_stds_lookup'

class ParameterType(BaseModel):
    description = CharField(null=True)
    parameter_type = CharField(primary_key=True)

    class Meta:
        db_table = 'parameter_type'

class AssayParameters(BaseModel):
    assay = ForeignKeyField(db_column='assay_id', rel_model=Assays, to_field='assay')
    assay_param = PrimaryKeyField(db_column='assay_param_id')
    parameter_type = ForeignKeyField(db_column='parameter_type', rel_model=ParameterType, to_field='parameter_type')
    parameter_value = CharField()

    class Meta:
        db_table = 'assay_parameters'

class AtcClassification(BaseModel):
    level1 = CharField(null=True)
    level1_description = CharField(null=True)
    level2 = CharField(null=True)
    level2_description = CharField(null=True)
    level3 = CharField(null=True)
    level3_description = CharField(null=True)
    level4 = CharField(null=True)
    level4_description = CharField(null=True)
    level5 = CharField(primary_key=True)
    who = CharField(db_column='who_id', null=True)
    who_name = CharField(null=True)

    class Meta:
        db_table = 'atc_classification'

class BindingSites(BaseModel):
    site = PrimaryKeyField(db_column='site_id')
    site_name = CharField(null=True)
    tid = ForeignKeyField(db_column='tid', null=True, rel_model=TargetDictionary, to_field='tid')

    class Meta:
        db_table = 'binding_sites'

class BioComponentSequences(BaseModel):
    component = PrimaryKeyField(db_column='component_id')
    component_type = CharField()
    description = CharField(null=True)
    organism = CharField(null=True)
    sequence = TextField(null=True)
    sequence_md5sum = CharField(null=True)
    tax = IntegerField(db_column='tax_id', null=True)

    class Meta:
        db_table = 'bio_component_sequences'

class Biotherapeutics(BaseModel):
    description = CharField(null=True)
    helm_notation = CharField(null=True)
    molregno = ForeignKeyField(db_column='molregno', primary_key=True, rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'biotherapeutics'

class BiotherapeuticComponents(BaseModel):
    biocomp = PrimaryKeyField(db_column='biocomp_id')
    component = ForeignKeyField(db_column='component_id', rel_model=BioComponentSequences, to_field='component')
    molregno = ForeignKeyField(db_column='molregno', rel_model=Biotherapeutics, to_field='molregno')

    class Meta:
        db_table = 'biotherapeutic_components'

class ProteinClassification(BaseModel):
    class_level = IntegerField()
    definition = CharField(null=True)
    parent = IntegerField(db_column='parent_id', null=True)
    pref_name = CharField(null=True)
    protein_class_desc = CharField()
    protein_class = PrimaryKeyField(db_column='protein_class_id')
    short_name = CharField(null=True)

    class Meta:
        db_table = 'protein_classification'

class ComponentSequences(BaseModel):
    accession = CharField(null=True, unique=True)
    component = PrimaryKeyField(db_column='component_id')
    component_type = CharField(null=True)
    db_source = CharField(null=True)
    db_version = CharField(null=True)
    description = CharField(null=True)
    organism = CharField(null=True)
    sequence = TextField(null=True)
    sequence_md5sum = CharField(null=True)
    tax = IntegerField(db_column='tax_id', null=True)

    class Meta:
        db_table = 'component_sequences'

class ComponentClass(BaseModel):
    comp_class = PrimaryKeyField(db_column='comp_class_id')
    component = ForeignKeyField(db_column='component_id', rel_model=ComponentSequences, to_field='component')
    protein_class = ForeignKeyField(db_column='protein_class_id', rel_model=ProteinClassification, to_field='protein_class')

    class Meta:
        db_table = 'component_class'

class Domains(BaseModel):
    domain_description = CharField(null=True)
    domain = PrimaryKeyField(db_column='domain_id')
    domain_name = CharField(null=True)
    domain_type = CharField()
    source_domain = CharField(db_column='source_domain_id')

    class Meta:
        db_table = 'domains'

class ComponentDomains(BaseModel):
    compd = PrimaryKeyField(db_column='compd_id')
    component = ForeignKeyField(db_column='component_id', rel_model=ComponentSequences, to_field='component')
    domain = ForeignKeyField(db_column='domain_id', null=True, rel_model=Domains, to_field='domain')
    end_position = IntegerField(null=True)
    start_position = IntegerField(null=True)

    class Meta:
        db_table = 'component_domains'

class ComponentSynonyms(BaseModel):
    component = ForeignKeyField(db_column='component_id', rel_model=ComponentSequences, to_field='component')
    component_synonym = CharField(null=True)
    compsyn = PrimaryKeyField(db_column='compsyn_id')
    syn_type = CharField(null=True)

    class Meta:
        db_table = 'component_synonyms'

class CompoundProperties(BaseModel):
    acd_logd = DecimalField(null=True)
    acd_logp = DecimalField(null=True)
    acd_most_apka = DecimalField(null=True)
    acd_most_bpka = DecimalField(null=True)
    alogp = DecimalField(index=True, null=True)
    aromatic_rings = IntegerField(null=True)
    full_molformula = CharField(null=True)
    full_mwt = DecimalField(null=True)
    hba = IntegerField(index=True, null=True)
    hba_lipinski = IntegerField(null=True)
    hbd = IntegerField(index=True, null=True)
    hbd_lipinski = IntegerField(null=True)
    heavy_atoms = IntegerField(null=True)
    med_chem_friendly = CharField(null=True)
    molecular_species = CharField(null=True)
    molregno = ForeignKeyField(db_column='molregno', primary_key=True, rel_model=MoleculeDictionary, to_field='molregno')
    mw_freebase = DecimalField(index=True, null=True)
    mw_monoisotopic = DecimalField(null=True)
    num_alerts = IntegerField(null=True)
    num_lipinski_ro5_violations = IntegerField(null=True)
    num_ro5_violations = IntegerField(index=True, null=True)
    psa = DecimalField(index=True, null=True)
    qed_weighted = DecimalField(null=True)
    ro3_pass = CharField(null=True)
    rtb = IntegerField(index=True, null=True)

    class Meta:
        db_table = 'compound_properties'

class StructuralAlertSets(BaseModel):
    alert_set = PrimaryKeyField(db_column='alert_set_id')
    priority = IntegerField()
    set_name = CharField(unique=True)

    class Meta:
        db_table = 'structural_alert_sets'

class StructuralAlerts(BaseModel):
    alert = PrimaryKeyField(db_column='alert_id')
    alert_name = CharField()
    alert_set = ForeignKeyField(db_column='alert_set_id', rel_model=StructuralAlertSets, to_field='alert_set')
    smarts = CharField()

    class Meta:
        db_table = 'structural_alerts'

class CompoundStructuralAlerts(BaseModel):
    alert = ForeignKeyField(db_column='alert_id', rel_model=StructuralAlerts, to_field='alert')
    cpd_str_alert = PrimaryKeyField(db_column='cpd_str_alert_id')
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'compound_structural_alerts'

class CompoundStructures(BaseModel):
    canonical_smiles = CharField(null=True)
    molfile = TextField(null=True)
    molregno = ForeignKeyField(db_column='molregno', primary_key=True, rel_model=MoleculeDictionary, to_field='molregno')
    standard_inchi = CharField(null=True)
    standard_inchi_key = CharField(index=True)

    class Meta:
        db_table = 'compound_structures'

class DefinedDailyDose(BaseModel):
    atc_code = ForeignKeyField(db_column='atc_code', rel_model=AtcClassification, to_field='level5')
    ddd_admr = CharField(null=True)
    ddd_comment = CharField(null=True)
    ddd = PrimaryKeyField(db_column='ddd_id')
    ddd_units = CharField(null=True)
    ddd_value = DecimalField(null=True)

    class Meta:
        db_table = 'defined_daily_dose'

class DrugMechanism(BaseModel):
    action_type = ForeignKeyField(db_column='action_type', null=True, rel_model=ActionType, to_field='action_type')
    binding_site_comment = CharField(null=True)
    direct_interaction = IntegerField(null=True)
    disease_efficacy = IntegerField(null=True)
    mec = PrimaryKeyField(db_column='mec_id')
    mechanism_comment = CharField(null=True)
    mechanism_of_action = CharField(null=True)
    molecular_mechanism = IntegerField(null=True)
    molregno = ForeignKeyField(db_column='molregno', null=True, rel_model=MoleculeDictionary, to_field='molregno')
    record = ForeignKeyField(db_column='record_id', rel_model=CompoundRecords, to_field='record')
    selectivity_comment = CharField(null=True)
    site = ForeignKeyField(db_column='site_id', null=True, rel_model=BindingSites, to_field='site')
    tid = ForeignKeyField(db_column='tid', null=True, rel_model=TargetDictionary, to_field='tid')

    class Meta:
        db_table = 'drug_mechanism'

class Products(BaseModel):
    ad_type = CharField(null=True)
    applicant_full_name = CharField(null=True)
    approval_date = DateField(null=True)
    black_box_warning = IntegerField(null=True)
    dosage_form = CharField(null=True)
    innovator_company = IntegerField(null=True)
    nda_type = CharField(null=True)
    oral = IntegerField(null=True)
    parenteral = IntegerField(null=True)
    product = CharField(db_column='product_id', primary_key=True)
    route = CharField(null=True)
    topical = IntegerField(null=True)
    trade_name = CharField(null=True)

    class Meta:
        db_table = 'products'

class Formulations(BaseModel):
    formulation = PrimaryKeyField(db_column='formulation_id')
    ingredient = CharField(null=True)
    molregno = ForeignKeyField(db_column='molregno', null=True, rel_model=MoleculeDictionary, to_field='molregno')
    product = ForeignKeyField(db_column='product_id', rel_model=Products, to_field='product')
    record = ForeignKeyField(db_column='record_id', rel_model=CompoundRecords, to_field='record')
    strength = CharField(null=True)

    class Meta:
        db_table = 'formulations'

class FracClassification(BaseModel):
    active_ingredient = CharField()
    frac_class = PrimaryKeyField(db_column='frac_class_id')
    frac_code = CharField()
    level1 = CharField()
    level1_description = CharField()
    level2 = CharField()
    level2_description = CharField(null=True)
    level3 = CharField()
    level3_description = CharField(null=True)
    level4 = CharField()
    level4_description = CharField(null=True)
    level5 = CharField(unique=True)

    class Meta:
        db_table = 'frac_classification'

class HracClassification(BaseModel):
    active_ingredient = CharField()
    hrac_class = PrimaryKeyField(db_column='hrac_class_id')
    hrac_code = CharField()
    level1 = CharField()
    level1_description = CharField()
    level2 = CharField()
    level2_description = CharField(null=True)
    level3 = CharField(unique=True)

    class Meta:
        db_table = 'hrac_classification'

class IracClassification(BaseModel):
    active_ingredient = CharField()
    irac_class = PrimaryKeyField(db_column='irac_class_id')
    irac_code = CharField()
    level1 = CharField()
    level1_description = CharField()
    level2 = CharField()
    level2_description = CharField()
    level3 = CharField()
    level3_description = CharField()
    level4 = CharField(unique=True)

    class Meta:
        db_table = 'irac_classification'

class LigandEff(BaseModel):
    activity = ForeignKeyField(db_column='activity_id', primary_key=True, rel_model=Activities, to_field='activity')
    bei = DecimalField(null=True)
    le = DecimalField(null=True)
    lle = DecimalField(null=True)
    sei = DecimalField(null=True)

    class Meta:
        db_table = 'ligand_eff'

class MechanismRefs(BaseModel):
    mec = ForeignKeyField(db_column='mec_id', rel_model=DrugMechanism, to_field='mec')
    mecref = PrimaryKeyField(db_column='mecref_id')
    ref = CharField(db_column='ref_id', null=True)
    ref_type = CharField()
    ref_url = CharField(null=True)

    class Meta:
        db_table = 'mechanism_refs'

class MoleculeAtcClassification(BaseModel):
    level5 = ForeignKeyField(db_column='level5', rel_model=AtcClassification, to_field='level5')
    mol_atc = PrimaryKeyField(db_column='mol_atc_id')
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'molecule_atc_classification'

class MoleculeFracClassification(BaseModel):
    frac_class = ForeignKeyField(db_column='frac_class_id', rel_model=FracClassification, to_field='frac_class')
    mol_frac = PrimaryKeyField(db_column='mol_frac_id')
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'molecule_frac_classification'

class MoleculeHierarchy(BaseModel):
    active_molregno = ForeignKeyField(db_column='active_molregno', null=True, rel_model=MoleculeDictionary, to_field='molregno')
    molregno = ForeignKeyField(db_column='molregno', primary_key=True, rel_model=MoleculeDictionary, related_name='molecule_dictionary_molregno_set', to_field='molregno')
    parent_molregno = ForeignKeyField(db_column='parent_molregno', null=True, rel_model=MoleculeDictionary, related_name='molecule_dictionary_parent_molregno_set', to_field='molregno')

    class Meta:
        db_table = 'molecule_hierarchy'

class MoleculeHracClassification(BaseModel):
    hrac_class = ForeignKeyField(db_column='hrac_class_id', rel_model=HracClassification, to_field='hrac_class')
    mol_hrac = PrimaryKeyField(db_column='mol_hrac_id')
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'molecule_hrac_classification'

class MoleculeIracClassification(BaseModel):
    irac_class = ForeignKeyField(db_column='irac_class_id', rel_model=IracClassification, to_field='irac_class')
    mol_irac = PrimaryKeyField(db_column='mol_irac_id')
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')

    class Meta:
        db_table = 'molecule_irac_classification'

class ResearchStem(BaseModel):
    res_stem = PrimaryKeyField(db_column='res_stem_id')
    research_stem = CharField(null=True, unique=True)

    class Meta:
        db_table = 'research_stem'

class MoleculeSynonyms(BaseModel):
    molregno = ForeignKeyField(db_column='molregno', rel_model=MoleculeDictionary, to_field='molregno')
    molsyn = PrimaryKeyField(db_column='molsyn_id')
    res_stem = ForeignKeyField(db_column='res_stem_id', null=True, rel_model=ResearchStem, to_field='res_stem')
    syn_type = CharField()
    synonyms = CharField(null=True)

    class Meta:
        db_table = 'molecule_synonyms'

class OrganismClass(BaseModel):
    l1 = CharField(null=True)
    l2 = CharField(null=True)
    l3 = CharField(null=True)
    oc = PrimaryKeyField(db_column='oc_id')
    tax = IntegerField(db_column='tax_id', null=True, unique=True)

    class Meta:
        db_table = 'organism_class'

class PatentUseCodes(BaseModel):
    definition = CharField()
    patent_use_code = CharField(primary_key=True)

    class Meta:
        db_table = 'patent_use_codes'

class PredictedBindingDomains(BaseModel):
    activity = ForeignKeyField(db_column='activity_id', null=True, rel_model=Activities, to_field='activity')
    confidence = CharField(null=True)
    predbind = PrimaryKeyField(db_column='predbind_id')
    prediction_method = CharField(null=True)
    site = ForeignKeyField(db_column='site_id', null=True, rel_model=BindingSites, to_field='site')

    class Meta:
        db_table = 'predicted_binding_domains'

class ProductPatents(BaseModel):
    delist_flag = IntegerField()
    drug_product_flag = IntegerField()
    drug_substance_flag = IntegerField()
    patent_expire_date = DateField()
    patent_no = CharField()
    patent_use_code = ForeignKeyField(db_column='patent_use_code', null=True, rel_model=PatentUseCodes, to_field='patent_use_code')
    prod_pat = PrimaryKeyField(db_column='prod_pat_id')
    product = ForeignKeyField(db_column='product_id', rel_model=Products, to_field='product')

    class Meta:
        db_table = 'product_patents'

class ProteinClassSynonyms(BaseModel):
    protclasssyn = PrimaryKeyField(db_column='protclasssyn_id')
    protein_class = ForeignKeyField(db_column='protein_class_id', rel_model=ProteinClassification, to_field='protein_class')
    protein_class_synonym = CharField(null=True)
    syn_type = CharField(null=True)

    class Meta:
        db_table = 'protein_class_synonyms'

class ProteinFamilyClassification(BaseModel):
    l1 = CharField()
    l2 = CharField(null=True)
    l3 = CharField(null=True)
    l4 = CharField(null=True)
    l5 = CharField(null=True)
    l6 = CharField(null=True)
    l7 = CharField(null=True)
    l8 = CharField(null=True)
    protein_class_desc = CharField()
    protein_class = PrimaryKeyField(db_column='protein_class_id')

    class Meta:
        db_table = 'protein_family_classification'

class ResearchCompanies(BaseModel):
    co_stem = PrimaryKeyField(db_column='co_stem_id')
    company = CharField(null=True)
    country = CharField(null=True)
    previous_company = CharField(null=True)
    res_stem = ForeignKeyField(db_column='res_stem_id', null=True, rel_model=ResearchStem, to_field='res_stem')

    class Meta:
        db_table = 'research_companies'

class SiteComponents(BaseModel):
    component = ForeignKeyField(db_column='component_id', null=True, rel_model=ComponentSequences, to_field='component')
    domain = ForeignKeyField(db_column='domain_id', null=True, rel_model=Domains, to_field='domain')
    site = ForeignKeyField(db_column='site_id', rel_model=BindingSites, to_field='site')
    site_residues = CharField(null=True)
    sitecomp = PrimaryKeyField(db_column='sitecomp_id')

    class Meta:
        db_table = 'site_components'

class TargetComponents(BaseModel):
    component = ForeignKeyField(db_column='component_id', rel_model=ComponentSequences, to_field='component')
    homologue = IntegerField()
    targcomp = PrimaryKeyField(db_column='targcomp_id')
    tid = ForeignKeyField(db_column='tid', rel_model=TargetDictionary, to_field='tid')

    class Meta:
        db_table = 'target_components'

class TargetRelations(BaseModel):
    related_tid = ForeignKeyField(db_column='related_tid', rel_model=TargetDictionary, to_field='tid')
    relationship = CharField()
    targrel = PrimaryKeyField(db_column='targrel_id')
    tid = ForeignKeyField(db_column='tid', rel_model=TargetDictionary, related_name='target_dictionary_tid_set', to_field='tid')

    class Meta:
        db_table = 'target_relations'

class UsanStems(BaseModel):
    annotation = CharField(null=True)
    major_class = CharField(null=True)
    stem = CharField()
    stem_class = CharField(null=True)
    subgroup = CharField()
    usan_stem = PrimaryKeyField(db_column='usan_stem_id')
    who_extra = IntegerField(null=True)

    class Meta:
        db_table = 'usan_stems'

class Version(BaseModel):
    comments = CharField(null=True)
    creation_date = DateField(null=True)
    name = CharField(primary_key=True)

    class Meta:
        db_table = 'version'

