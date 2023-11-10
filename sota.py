import os

score_list = []

base_path = "result/KoBERT-l2-d2_1"
for model_name in os.listdir("result/KoBERT-l2-d2_1"):
    score_txt_path = os.path.join(base_path, model_name, "scores.txt")
    try:
        with open(score_txt_path, mode="r") as f:
            texts = f.readlines()
            # score = texts[-1].split(" : ")[-1]
            # print(texts[1], texts[5], texts[9])

            p1_score = float(texts[1].split()[-1])
            p2_score = float(texts[6].split()[-1])
            p3_score = float(texts[11].split()[-1])
            score = p1_score + p2_score + p3_score
            # print(score)
            score_list.append((model_name, score, p1_score, p2_score, p3_score))
    except Exception as e:
        print(e)

score_list.sort(key=lambda x: x[1], reverse=True)
for score in score_list[:10]:
    print(score)
