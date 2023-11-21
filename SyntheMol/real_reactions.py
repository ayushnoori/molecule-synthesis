"""SMARTS representations of the most common REAL reactions.
Reference: https://docs.google.com/document/d/1LDgRXf4P-uOXQEmgJPgVhuOK32I2u0FXSB8tzDenN6U/edit?usp=sharing
"""
from reactions import QueryMol, Reaction

REAL_REACTIONS = [
    Reaction(
        reagents=[
            QueryMol('CC(C)(C)OC(=O)[N:1]([*:2])[*:3].[*:4][N:5]([H])[*:6]'),
            QueryMol('[OH1][C:7]([*:8])=[O:9]'),
            QueryMol('[OH1][C:10]([*:11])=[O:12]')
        ],
        product=QueryMol('[*:4][N:5]([*:6])[C:7](=[O:9])[*:8].[*:3][N:1]([*:2])[C:10](=[O:12])[*:11]'),
        reaction_id=275592
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[OH1][C:4]([*:5])=[O:6]')
        ],
        product=QueryMol('[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]'),
        reaction_id=22
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[OH1][C:4]([*:5])=[O:6]')
        ],
        product=QueryMol('[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]'),
        reaction_id=11
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[OH1][C:4]([*:5])=[O:6]')
        ],
        product=QueryMol('[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]'),
        reaction_id=527
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[H:3]'),
            QueryMol('[*:4][N:5]([H])[*:6]')
        ],
        product=QueryMol('O=C([N:2]([*:1])[H:3])[N:5]([*:4])[*:6]'),
        reaction_id=2430
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[H:3]'),
            QueryMol('[*:4][N:5]([H])[H:6]')
        ],
        product=QueryMol('O=C([N:2]([*:1])[H:3])[N:5]([*:4])[H:6]'),
        reaction_id=2708
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[OH1][C:4]([*:5])=[O:6]')
        ],
        product=QueryMol('[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]'),
        reaction_id=240690
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[F,Cl,Br,I][*:4]')
        ],
        product=QueryMol('[*:1][N:2]([*:3])[*:4]'),
        reaction_id=2230
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[H:3]'),
            QueryMol('[*:4][N:5]([H])[H:6]')
        ],
        product=QueryMol('O=C(C(=O)[N:2]([*:1])[H:3])[N:5]([*:4])[H:6]'),
        reaction_id=2718
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[O:4]=[S:5](=[O:6])([F,Cl,Br,I])[*:7]')
        ],
        product=QueryMol('[O:4]=[S:5](=[O:6])([*:7])[N:2]([*:1])[*:3]'),
        reaction_id=40
    ),
    Reaction(
        reagents=[
            QueryMol('[OH1:1][C:2]([*:3])=[O:4]'),
            QueryMol('[F,Cl,Br,I][*:5]')
        ],
        product=QueryMol('[O:4]=[C:2]([*:3])[O:1][*:5]'),
        reaction_id=1458
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[*:4][N:5]([H])[H:6]')
        ],
        product=QueryMol('O=C(C(=O)[N:2]([*:1])[*:3])[N:5]([*:4])[H:6]'),
        reaction_id=271948
    ),
    Reaction(
        reagents=[
            QueryMol('[*:1][N:2]([H])[*:3]'),
            QueryMol('[F,Cl,Br,I][*:4]')
        ],
        product=QueryMol('[*:1][N:2]([*:3])[*:4]'),
        reaction_id=27
    )
]


if __name__ == '__main__':
    # Save reaction SMARTS and reaction IDs in a CSV file
    import pandas as pd

    smarts, reaction_ids = [], []

    for reaction in REAL_REACTIONS:
        smarts.append(
            f'{".".join(f"({reagent.smarts_with_atom_mapping})" for reagent in reaction.reagents)}'
            f'>>({reaction.product.smarts_with_atom_mapping})'
        )
        reaction_ids.append(reaction.id)

    df = pd.DataFrame({'smarts': smarts, 'reaction_id': reaction_ids})
    df.to_csv('data/real_reaction_smarts.csv', index=False)
