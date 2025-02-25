import pandas as pd
import utils
from inference_model_based_on_sub_obj import inference
compute_instance = utils.Compute()
def remove_periods(input_string):
    return input_string.replace('.', '')
class Reasoner:
    def __init__(self, dataframe):

        self.dataframe = dataframe

    @staticmethod
    def is_closed_causal_loop(claim_ERE,answer_ERE,R_S,R_O):



        A, B, C, D = claim_ERE[0],claim_ERE[2],answer_ERE[0],answer_ERE[2]


        a_b_relation = claim_ERE[1]
        c_d_relation = answer_ERE[1]
        a_c_relation = R_S
        d_b_relation = R_O


        if a_b_relation in ['enable', 'intend','cause']:
            if c_d_relation in ['enable', 'intend','cause']and a_c_relation in ['enable', 'intend','cause']and d_b_relation in ['enable', 'intend','cause']:
                return True
            elif  c_d_relation =='prevent' and a_c_relation =='prevent' and d_b_relation in['enable', 'intend','cause']:
                return True
            elif  c_d_relation =='prevent' and a_c_relation =='cause' and d_b_relation == 'prevent':
                return True
        else:
            if c_d_relation  in ['enable', 'intend','cause'] and a_c_relation in ['enable', 'intend','cause']and d_b_relation =='prevent':
                return True

    # def apply_rules(self):
    #
    #     results = []
    #     for _, row in self.dataframe.iterrows():
    #         verdict = self.rule_check(row,'PLM')
    #         results.append(verdict)
    #
    #     self.dataframe['Prediction_Verdict'] = results
    #     return self.dataframe

    def rule_check(self,  model):
     # self.dataframe['Prediction_Verdict'] = None
     logically_misaligned = set()
     logically_aligned = set()


     for idx, row in self.dataframe.iterrows():
        # if pd.notna(row['Prediction_Verdict']):
        #      print('not none verdict',row['Prediction_Verdict'] )
        #      continue

        claim_ere = eval(row['claim_ERE'])
        answer_ere = eval(row['answer_ERE'])
        if model == 'PLM':
            print('I am in PLM case')
            R_S = inference.predict_relation(claim_ere[0], answer_ere[0])
            R_O = inference.predict_relation(answer_ere[2], claim_ere[2])

        else:
            print('I am in LLMs case')
            R_S=remove_periods(row['relation_subject'])
            R_O=remove_periods(row['relation_object'])



        sim_context_event_sub = row['sim_context+event_sub']
        sim_context_event_obj = row['sim_context+event_obj']
        polarity_C_A_s = row['pol_context+event_sub']
        polarity_C_A_o = row['pol_context+event_object']
        # relation_claim_answer = row['relation_claim_answer']
        grouped = self.dataframe.groupby('Claim')

        sub_claim = claim_ere[0]
        sub_answer = answer_ere[0]
        object_claim = claim_ere[2]
        object_answer = answer_ere[2]
        r_claim = claim_ere[1]
        r_answer = answer_ere[1]

        # Set to track claims identified as cherrypicking
        cherrypicked_claims = set()

        # Check for cherry-picking conditions (if any claim has been cherry-picked)
        for claim, group in grouped:
            if len(group) < 2:
                continue

            found_cherrypicking = False

            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i >= j:
                        continue

                    sub1, rel1, obj1 = eval(row1['answer_ERE'])
                    sub2, rel2, obj2 = eval(row2['answer_ERE'])

                    obj_similarity = compute_instance.compute_similarity(obj1, obj2)
                    _, obj_polarity1, _ = compute_instance.analyze_claim_answer(obj1, obj2)

                    sub_similarity = compute_instance.compute_similarity(sub1, sub2)
                    _, sub_polarity1, _ = compute_instance.analyze_claim_answer(sub1, sub2)
                    _, polarity, _ = compute_instance.analyze_claim_answer(str(eval(row1['answer_ERE'])), str(eval(row2['answer_ERE'])))


                    # Check for cherry-picking conditions
                    if (
                            (obj_similarity > 0.54  and
                             sub_similarity < 0.54  and rel1 == rel2)


                            or (sub_similarity > 0.54  and
                                obj_similarity < 0.54 and rel1 == rel2 )
                    ):
                        found_cherrypicking = True
                        break

                if found_cherrypicking:
                    cherrypicked_claims.add(claim)

                    break

        self.dataframe.loc[self.dataframe['Claim'].isin(
            cherrypicked_claims), 'Prediction_Verdict'] = 'Conflicting Evidence/Cherrypicking'

        if reasoner.is_closed_causal_loop(claim_ere, answer_ere,R_S,R_O):
            print('found closed loop')
            print(claim)
            logically_aligned.add(claim)
            verdict= 'Supported'
            # self.dataframe.loc[self.dataframe['Claim'].isin(
            #     logically_aligned), 'Prediction_Verdict'] = verdict
            self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
            continue



        polarity_C_A = row['pol_claim_answer']
        similarity_sub_sub,_,_=compute_instance.analyze_claim_answer(str(claim_ere[0]), str(answer_ere[0]))
        similarity_obj_obj, _, _ = compute_instance.analyze_claim_answer(str(claim_ere[2]), str(answer_ere[2]))
        #checking logical inconsistency

        if (claim_ere[1] == answer_ere[1]) or (
                claim_ere[1] in ['cause', 'intend', 'enable'] and answer_ere[1] in ['cause', 'intend', 'enable']):
            if (sim_context_event_sub > 0.54  or similarity_sub_sub>0.54 )and (sim_context_event_obj > 0.5 or similarity_obj_obj>0.54) and polarity_C_A == 'PN':
                print('found logically_misaligned')
                logically_misaligned.add(claim)
                print(logically_misaligned)

                verdict='Refuted'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue



            #checking logical consistency
            if (((sim_context_event_sub > 0.54 or similarity_sub_sub>0.54) and (sim_context_event_obj > 0.54 or similarity_obj_obj>0.54)) or (
                    claim_ere[0] == answer_ere[0] and answer_ere[2] == claim_ere[2]) ) and polarity_C_A in ['NN', 'PP'] :
                print('found aligned')
                logically_aligned.add(claim)
                verdict = 'Supported'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue

            #check consisteny with causation
            elif(( R_S in ['cause',  'enable', 'intend']  and (sim_context_event_obj > 0.54 or similarity_obj_obj>0.54)) or (R_O in ['cause',  'enable', 'intend']  and (sim_context_event_sub > 0.54 or similarity_sub_sub>0.54) ) and polarity_C_A in ['NN',
                                                                                                                'PP']) :
                print('found causal alignement')
                logically_aligned.add(claim)
                verdict = 'Supported'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue

            # elif sim_context_event_sub > 0.54 and polarity_C_A in ['NN',
            #                                                                'PP'] and sim_context_event_obj > 0.54 and polarity_C_A == 'PN':
            #     return 'Refuted'
            elif (sim_context_event_sub < 0.54 or similarity_sub_sub<0.54) and (sim_context_event_obj > 0.54 or similarity_obj_obj> 0.54) and polarity_C_A in ['NN', 'PP']:
                logically_misaligned.add(claim)
                print('also misaligned')
                verdict = 'Refuted'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue
            elif sim_context_event_sub > 0.54 and sim_context_event_obj > 0.54 and polarity_C_A == 'PN':
                logically_misaligned.add(claim)
                print('again misaligned')
                verdict = 'Refuted'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue

        if claim_ere[1] != answer_ere[1]:
            if  (sim_context_event_sub > 0.54 or similarity_sub_sub>0.54) and claim_ere[1] in ['cause', 'intend', 'enable'] and answer_ere=='prevent' and R_O=='prevent':
                logically_aligned.add(claim)
                print('aligned')
                verdict = 'Supported'
                self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                continue
            if (claim_ere[1] in ['cause', 'intend', 'enable'] and answer_ere[1] in ['prevent']) or (
                    answer_ere[1] in ['cause', 'intend', 'enable'] and claim_ere[1] in ['prevent']):
                if (sim_context_event_sub > 0.54 or similarity_sub_sub>0.54) and (sim_context_event_obj > 0.54 or similarity_obj_obj > 0.54 )and polarity_C_A in ['NN',
                                                                                                             'PP'] :
                    logically_misaligned.add(claim)
                    print('not aligned')
                    verdict = 'Refuted'
                    self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                    continue
                elif (sim_context_event_sub > 0.54 or similarity_sub_sub>0.54)  and (sim_context_event_obj > 0.54 or similarity_obj_obj > 0.54 ) and polarity_C_A == 'PN':
                    logically_aligned.add(claim)
                    print('not aligned also')
                    verdict = 'Supported'
                    self.dataframe.at[idx, 'Prediction_Verdict'] = verdict
                    continue


        continue
     print(logically_misaligned)
     print(logically_aligned)
     return self.dataframe


    def compare_with_label(self):

        self.dataframe['result'] = self.dataframe.apply(
            lambda row: 1 if row['Prediction_Verdict'] is not None and row['Prediction_Verdict'] == row['Label'] else 0,
            axis=1
        )


    def calculate_accuracy(self):

        filtered_df = self.dataframe[self.dataframe['Prediction_Verdict'].notna()]
        if len(filtered_df) == 0:
            return 0
        accuracy1 = filtered_df['result'].mean()
        accuracy_complete= self.dataframe['result'].mean()
        return accuracy1, accuracy_complete


if __name__ == "__main__":

    df = pd.read_csv('/data/Youss/Fact_cheking/reasoner/inference_LLMs/input_file.csv')


    reasoner = Reasoner(df)


    updated_df = reasoner.rule_check('PLM')
    non_none_count = df['Prediction_Verdict'].notna().sum()
    print(non_none_count)


    reasoner.compare_with_label()
    accuracy1, accuracy_complete = reasoner.calculate_accuracy()


    updated_df.to_csv('PLM.csv', index=False)
    print(f"Accuracy of non-None predictions: {accuracy1:.2%}")
    print(f"Accuracy of all predictions: {accuracy_complete:.2%}")
