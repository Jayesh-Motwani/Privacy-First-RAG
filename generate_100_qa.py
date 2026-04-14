import json
import random

random.seed(42)
domains = ['domestic_violence', 'child_protection', 'marriage_and_divorce', 'maintenance', 'adoption']
complexities = ['simple', 'complex', 'cross-act']
demographics = ['married_woman', 'minor', 'divorced_woman', 'any']

domain_data = {
    'domestic_violence': {
        'acts': ['Protection of Women from Domestic Violence Act, 2005'],
        'scenarios': [
            ('My husband beats me', '18', 'Protection orders', 'simple', ['Section 18 protection orders']),
            ('I am being thrown out of my house', '19', 'Residence orders', 'complex', ['Section 19 residence orders']),
            ('Need medical expenses for injury', '20', 'Monetary reliefs', 'simple', ['Section 20 allows monetary relief']),
            ('Can I get custody of my child?', '21', 'Custody orders', 'complex', ['Section 21 allows temporary custody']),
            ('Pending divorce and domestic violence', '17', 'Right to reside in a shared household', 'cross-act', ['Section 17 protects residence right', 'Conflict: HMA_Section_13 vs DV_Act_Section_17'])
        ]
    },
    'child_protection': {
        'acts': ['Protection of Children from Sexual Offences Act, 2012', 'Juvenile Justice (Care and Protection of Children) Act, 2015'],
        'scenarios': [
            ('Teacher touched me badly', '19', 'Reporting of offences', 'complex', ['Incident must be reported to SJPU']),
            ('Media revealed child name in court', '23', 'Procedure for media', 'simple', ['Section 23 prohibits disclosure']),
            ('Bail for a serious POCSO offender', '29', 'Presumption of guilt', 'cross-act', ['POCSO Section 29 presumption vs CrPC Section 167', 'Conflict: POCSO_Section_29 vs CrPC_Section_167'])
        ]
    },
    'marriage_and_divorce': {
        'acts': ['Hindu Marriage Act, 1955'],
        'scenarios': [
            ('Spouse has an affair', '13', 'Divorce', 'simple', ['Adultery is a ground for divorce']),
            ('Husband forces me to return home', '9', 'Restitution of conjugal rights', 'cross-act', ['HMA Section 9 vs DV Act Section 18', 'Conflict: HMA_Section_9 vs DV_Act_Section_18']),
            ('Is a child born out of void marriage legitimate?', '16', 'Legitimacy of children', 'complex', ['Considered legitimate']),
            ('Divorce by mutual consent', '13B', 'Divorce by mutual consent', 'complex', ['Requires 1 year of separation'])
        ]
    },
    'maintenance': {
        'acts': ['The Code of Criminal Procedure, 1973', 'Hindu Adoptions and Maintenance Act, 1956'],
        'scenarios': [
            ('Claiming maintenance from father', '125', 'Order for maintenance of wives, children and parents', 'simple', ['CrPC Section 125 allows maintenance claim']),
            ('Maintenance under HAMA and CrPC', '18', 'Maintenance of wife', 'cross-act', ['HAMA Section 18 vs CrPC Section 125', 'Conflict: HAMA_Section_18 vs CrPC_Section_125']),
            ('Alimony pendente lite', '24', 'Maintenance pendente lite', 'simple', ['HMA Section 24 allows maintenance during pendency'])
        ]
    },
    'adoption': {
        'acts': ['Juvenile Justice (Care and Protection of Children) Act, 2015', 'Hindu Adoptions and Maintenance Act, 1956'],
        'scenarios': [
            ('Hindu male adopting a Christian child', '56', 'Adoption', 'cross-act', ['JJ Act allows secular adoption', 'Conflict: HAMA_Section_6 vs JJ_Act_Section_56']),
            ('Adopting a child of same gender', '11', 'Other conditions for a valid adoption', 'simple', ['HAMA restricts adopting a son if one already exists']),
            ('Natural guardian vs JJ Act adoption', '56', 'Adoption', 'cross-act', ['Child welfare principal under JJ Act prevails', 'Conflict: Guardianship_Act_Section_6 vs JJ_Act_Section_56'])
        ]
    }
}

dataset = []
for i in range(1, 101):
    domain = domains[i % len(domains)]
    demo = demographics[i % len(demographics)]
    scenario = random.choice(domain_data[domain]['scenarios'])
    
    query_text, sec_num, sec_title, comp, key_points = scenario
    
    conflicts = []
    if comp == 'cross-act' and 'Conflict:' in key_points[-1]:
        conflicts.append(key_points[-1].replace('Conflict: ', ''))
        kp_clean = key_points[:-1]
    else:
        kp_clean = key_points
        
    acts = domain_data[domain]['acts']
    act_name = random.choice(acts)

    dataset.append({
        'id': f'Q{i:03d}',
        'raw_query': f'{query_text}. What is the law regarding this?',
        'expected_rewritten_query': f'What are the provisions regarding {sec_title} under the relevant acts?',
        'gold_passages': [{
            'act_name': act_name,
            'section_number': sec_num,
            'section_title': sec_title
        }],
        'gold_answer_key_points': kp_clean,
        'expected_conflicts': conflicts,
        'user_profile': {
            'jurisdiction': 'central',
            'personal_law': 'hindu' if domain in ['marriage_and_divorce', 'adoption'] else 'any',
            'demographic': demo
        },
        'complexity': comp,
        'domain': domain
    })

with open('benchmark/gold_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f'Successfully generated {len(dataset)} QA pairs to benchmark/gold_dataset.json')
