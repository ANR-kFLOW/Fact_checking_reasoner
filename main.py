from utils import Compute
import pandas as pd
if __name__ == "__main__":
    compute = Compute()
    # model_path = '/data/Youss/Fact_cheking/AVeriTeC/ERE_models/best_model_st1.pt'
    # rebel_model_path = '/data/Youss/Fact_cheking/AVeriTeC/ERE_models/augmneted_model.pth'
    data= pd.read_csv('/data/Youss/Fact_cheking/reasoner/manually_ERE.csv')


    data['sim_claim_answer'] = data.apply(
        lambda row: compute.compute_similarity(row['Claim'], row['Answer']), axis=1
    )

    data['sim_context+event_sub'] = data.apply(
        lambda row: compute.compute_similarity(
            row['Claim'] + " " + str(compute.extract_events_R(row['claim_ERE'])[0]), row['Answer'] + " " + str(compute.extract_events_R(row['answer_ERE'])[0])
        ), axis=1
    )

    data['sim_context+event_obj'] = data.apply(
        lambda row: compute.compute_similarity(
            row['Claim'] + " " + str(compute.extract_events_R(row['claim_ERE'])[2]), row['Answer'] + " " + str(compute.extract_events_R(row['answer_ERE'])[2])
        ), axis=1
    )

    # data['pol_context+event_sub'] = data.apply(
    #     lambda row: compute.analyze_claim_answer(
    #        row['claim_ERE'], row['answer_ERE'])
    #     )[1], axis=1

    data['pol_context+event_sub'] = data.apply(
        lambda row: compute.analyze_claim_answer(compute.extract_events_R(row['claim_ERE'])[0], compute.extract_events_R(row['answer_ERE'])[0])[1],
        axis=1
    )
    data['pol_claim_answer'] = data.apply(
        lambda row: compute.analyze_claim_answer(row['claim_ERE'], row['answer_ERE'])[1],
        axis=1
    )
    #
    # data['pol_context+event_object'] = data.apply(
    #     lambda row: compute.analyze_claim_answer(
    #        row['claim_ERE'], row['answer_ERE'])
    #     )[1], axis=1
    data['pol_context+event_object'] = data.apply(
        lambda row: compute.analyze_claim_answer(compute.extract_events_R(row['claim_ERE'])[2],
                                                 compute.extract_events_R(row['answer_ERE'])[2])[1],
        axis=1
    )

    # data['pearson_context+event_sub'] = data.apply(
    #     lambda row: compute.compute_pearson_correlation(
    #         row['Claim'] + " " + row['claim_ERE'], row['Answer'] + " " + row['answer_ERE']
    #     ), axis=1
    # )
    #
    # data['pearson_context+event_obj'] = data.apply(
    #     lambda row: compute.compute_pearson_correlation(
    #         row['Claim'] + " " + row['claim_ERE'], row['Answer'] + " " + row['answer_ERE']
    #     ), axis=1
    # )
    # data['relation_claim_answer'] = data.apply(
    #     lambda row: compute.perform_ere_on_text(row['Claim'] + " " + row['Answer'], model_path, rebel_model_path
    #                                             ), axis=1
    # )

    data.to_csv("manually_ERE_sim_pol.csv", index=False)

