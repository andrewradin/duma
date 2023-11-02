
/**
 * This file acts as the main interface between our Javascript code and our Django HTML templates.
 * 
 * Any functions or classes that you want accessible from template should be 'exported' here by assigning
 * to a window property.
 */
import {drawPathwayTree} from './pathway_tree';
import {drawMolSearch, drawMultiInput, ProtAndDirSelect, MolSearch, ProtSearch} from './mol_search';
import {addDataTableFilters, makeDataTableMerger} from './datatable_filter';

window.drawPathwayTree = drawPathwayTree;
window.drawMolSearch = drawMolSearch;
window.drawMultiInput = drawMultiInput;
window.ProtSearch = ProtSearch;
window.ProtAndDirSelect = ProtAndDirSelect;
window.MolSearch = MolSearch;

window.addDataTableFilters = addDataTableFilters;
window.makeDataTableMerger = makeDataTableMerger;