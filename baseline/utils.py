def retriever_prec_k(topk_list, retrieved_df):
        result_dict = {}
        count = [0]*len(topk_list)

        # iterate through each passage+query pair
        for ind, _ in enumerate(range(len(retrieved_df))):
            contexts = retrieved_df['context'][ind].split('<SEP>')
            gold_answer = retrieved_df['original_context'][ind]
            for order, k in enumerate(topk_list):
                if gold_answer in contexts[:k]: 
                    count[order] += 1
        # print(count)

        # compute precision at each k
        for ind, k in enumerate(topk_list):        
            result_dict[f'P@{k}'] = f'{round(count[ind]/len(df)*100,1)}%'

        return result_dict