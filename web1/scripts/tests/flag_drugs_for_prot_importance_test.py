


from pytest import approx

def test_flag_drugs():
    from scripts.flag_drugs_for_prot_importance import compute_target_scores, find_drugs_to_flag, make_flag_str


    wsa_weighted_scores = {
            1001: {
                'literature_otarg_glee_depend_codesmax': 0.5,
                'knowndrug_otarg_sigdif_codes_absdir': 0.25,
                'defus_wzs': 0.25
            },
            1002: {
                'defus_wzs': 1.0
            },
            1003: {
                'defus_wzs': 0.25,
                'pathsum_direct': 0.75
            }
        }


    piece_data = [
            [1001, 'literature_otarg_glee_depend', {
                'codesMax': {
                    'P001': 0.2,
                    'P003': 0.8
                },
                'codesMin': {
                    'P001': 1.0
                }
            }],
            [1001, 'knowndrug_otarg_sigdif_codes', {
                'absDir': {
                    'P002': 1.0
                    }
            }],
            [1001, 'defus', {
                'wzs': {
                    'P004': 1.0
                }
            }],
            [1002, 'defus', {
                'wzs': {
                    'P004': 0.3,
                    'P001': 0.7
                },
            }],
            [1003, 'defus', {
                'wzs': {
                    'P002': 0.1,
                    'P003': 0.9
                }
            }],
            [1003, 'pathsum', {
                'direct': {
                    'P001': 0.4,
                    'P002': 0.6
                }
            }],
        ]
    expected_scores = {
            1001: {
                'P001': 0.1,
                'P002': 0.25,
                'P003': 0.4,
                'P004': 0.25
                },
            1002: {
                'P001': 0.7,
                'P004': 0.3
                },
            1003: {
                'P001': 0.3,
                'P002': 0.475,
                'P003': 0.225
                }
            }
    expected_flags = [1001, 1002, 1003]
    all_uniprots = ["P001", "P002", "P003", "P004"]
    target_scores = compute_target_scores(all_uniprots, piece_data, wsa_weighted_scores)

    assert list(expected_scores.keys()) == list(target_scores.keys())
    for wsa in expected_scores:
        assert expected_scores[wsa] == approx(target_scores[wsa])

    drugs_to_flag = find_drugs_to_flag(target_scores, 0.4)
    assert sorted(drugs_to_flag) == sorted(expected_flags)

    gene_map = {'P001': 'gene1', 'P002': 'gene2', 'P003': 'gene3', 'P004': 'gene4'}
    flag_str = make_flag_str(1002, target_scores, gene_map)
    assert flag_str == "gene1=0.70 gene4=0.30"


    # Try with a limited set of uniprots
    uniprots2 = ['P002', 'P003']
    target_scores = compute_target_scores(uniprots2, piece_data, wsa_weighted_scores)
    drugs_to_flag = find_drugs_to_flag(target_scores, 0.65)
    assert sorted(drugs_to_flag) == [1001, 1003]
    flag_str = make_flag_str(1001, target_scores, gene_map)
    assert flag_str == "gene3=0.40 gene2=0.25"


    

    

