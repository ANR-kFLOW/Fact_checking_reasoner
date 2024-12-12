import pandas as pd
import utils
from inference_model_based_on_sub_obj import inference
compute_instance = utils.Compute()
class Reasoner:
    def __init__(self, dataframe):

        self.dataframe = dataframe

    @staticmethod
    def is_closed_causal_loop(claim_ERE,answer_ERE):

        A, B, C, D = claim_ERE[0],claim_ERE[2],answer_ERE[0],answer_ERE[2]


        a_b_relation = claim_ERE[1]
        c_d_relation = answer_ERE[1]
        a_c_relation = inference.predict_relation(A, C)
        d_b_relation = inference.predict_relation(D, B)


        if a_b_relation in ['enable, intend,cause']:
            if c_d_relation in ['enable', 'intend','cause']and a_c_relation in ['enable', 'intend','cause']and d_b_relation in ['enable', 'intend','cause']:
                return True
            elif  c_d_relation =='prevent' and a_c_relation =='prevent' and d_b_relation in['enable', 'intend','cause']:
                return True
            elif  c_d_relation =='prevent' and a_c_relation =='cause' and d_b_relation == 'prevent':
                return True
        else:
            if c_d_relation  in ['enable', 'intend','cause'] and a_c_relation in ['enable', 'intend','cause']and d_b_relation =='prevent':
                return True





    def apply_rules(self):

        results = []
        for _, row in self.dataframe.iterrows():

            verdict = self.rule_check(row)
            results.append(verdict)

        self.dataframe['Prediction_Verdict'] = results
        return self.dataframe

    def rule_check(self, row):

        claim_ere = row['claim_ERE'].split()
        answer_ere = row['answer_ERE'].split()

        sim_context_event_sub = row['sim_context+event_sub']
        sim_context_event_obj = row['sim_context+event_obj']
        pol_context_event_sub = row['pol_context+event_sub']
        pol_context_event_obj = row['pol_context+event_object']
        relation_claim_answer = row['relation_claim_answer']
        grouped = self.dataframe.groupby('Claim')
        #Here I am checking the chains of causality
        sub_claim=claim_ere[0]
        sub_answer=answer_ere[0]
        object_claim=claim_ere[2]
        object_answer= answer_ere[2]
        r_claim=claim_ere[1]
        r_answer=answer_ere[1]

        if  reasoner.is_closed_causal_loop(claim_ere,answer_ere):
            return 'Supported'
        if inference.predict_relation(sub_claim,sub_answer) in ['enable', 'intend','cause'] and inference.predict_relation(object_answer,object_claim) in ['enable', 'intend','cause'] and r_claim in ['enable', 'intend','cause'] and r_answer in['enable', 'intend','cause']:
            return 'Supported'
        elif inference.predict_relation(sub_claim,sub_answer) ==  inference.predict_relation(object_answer,object_claim) and r_claim==r_answer:
            return 'Supported'


        #in this part I am checking the cherry-picking scenarios
        for claim, group in grouped:
            if len(group) < 2:
                continue

            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i >= j:
                        continue
                    # print(row1['answer_ERE'].split())

                    sub1, rel1, obj1 = eval(row1['answer_ERE'])
                    sub2, rel2, obj2 = eval(row2['answer_ERE'])


                    obj_similarity = compute_instance.compute_similarity(obj1,obj2)
                    _,obj_polarity1,_ =compute_instance.analyze_claim_answer(obj1, obj2)



                    sub_similarity = compute_instance.compute_similarity(sub1,sub2)
                    _,sub_polarity1,_ =compute_instance.analyze_claim_answer(sub1, sub2)



                    if (
                            obj_similarity > 0.6
                            and obj_polarity1 == 'PN'

                            and sub_similarity > 0.6
                            and sub_polarity1 in ['NN', 'PP']

                            and rel1 == rel2
                    ) or ( obj_similarity > 0.6
                            and obj_polarity1 == ['NN', 'PP']

                            and sub_similarity < 0.6


                            and rel1 == rel2):
                        self.dataframe.loc[(self.dataframe[
                                                'Claim'] == claim), 'Prediction_Verdict'] = 'Conflicting Evidence/Cherrypicking'
                        break
        if (claim_ere[1] == answer_ere[1]) or (
                claim_ere[1] in ['cause', 'intend', 'enable'] and answer_ere[1] in ['cause', 'intend', 'enable']):
            # if polarity_object_subject_claim == 'PN' and similarity_object_subject_claim > 0.6:
            #     return 'Refuted'
            if sim_context_event_sub > 0.6 and sim_context_event_obj > 0.6 and pol_context_event_sub in ['NN',
                                                                                                         'PP'] and pol_context_event_obj in [
                'NN', 'PP']:
                return 'Supported'
            elif relation_claim_answer == 'cause' and sim_context_event_obj > 0.6 and pol_context_event_sub in ['NN',
                                                                                                                'PP'] and pol_context_event_obj in [
                'NN', 'PP']:
                return 'Supported'
            elif sim_context_event_sub > 0.6 and sim_context_event_obj > 0.6 and pol_context_event_obj == 'PN':
                return 'Refuted'

        if sim_context_event_sub < 0.6 and sim_context_event_obj > 0.6 and pol_context_event_obj in ['NN', 'PP']:
            return 'Refuted'

        if sim_context_event_sub > 0.6 and sim_context_event_obj > 0.6 and pol_context_event_sub in ['NN',
                                                                                                     'PP'] and pol_context_event_obj == 'PN':
            return 'Refuted'
        if claim_ere[1] != answer_ere[1]:
            if (claim_ere[1] in ['cause', 'intend', 'enable'] and answer_ere[1] in ['prevent']) or (
                    answer_ere[1] in ['cause', 'intend', 'enable'] and claim_ere[1] in ['prevent']):
                if sim_context_event_sub > 0.6 and sim_context_event_obj > 0.6 and pol_context_event_sub in ['NN',
                                                                                                             'PP'] and pol_context_event_obj in [
                    'NN', 'PP']:
                    return 'Refuted'
                elif sim_context_event_sub > 0.6 and sim_context_event_obj > 0.6 and pol_context_event_obj == 'PN':
                    return 'Supported'



        return None


    def compare_with_label(self):

        self.dataframe['result'] = self.dataframe.apply(
            lambda row: 1 if row['Prediction_Verdict'] is not None and row['Prediction_Verdict'] == row['Label'] else 0,
            axis=1
        )


    def calculate_accuracy(self):

        filtered_df = self.dataframe[self.dataframe['Prediction_Verdict'].notna()]
        if len(filtered_df) == 0:
            return 0
        accuracy = filtered_df['result'].mean()
        return accuracy


if __name__ == "__main__":

    df = pd.read_csv('use_cases_with_sim_pol_pear.csv')


    reasoner = Reasoner(df)


    updated_df = reasoner.apply_rules()
    reasoner.compare_with_label()
    accuracy = reasoner.calculate_accuracy()


    updated_df.to_csv('output_file_2.csv', index=False)
    print(f"Accuracy of non-None predictions: {accuracy:.2%}")