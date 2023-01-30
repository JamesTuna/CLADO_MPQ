import torch

TRAIN_MODEL_PATH = "/mnt/scratch/model_in_do_train.pt"
EVAL_MODEL_PATH = "/mnt/scratch/model_in_do_eval.pt"

model_train = torch.load(TRAIN_MODEL_PATH)
model_eval = torch.load(EVAL_MODEL_PATH)


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception


# layer_names = [
#     "attention.self.query",
#     "attention.self.key",
#     "attention.self.value",
#     "attention.output.dense",
#     "intermediate.dense",
#     "output.dense",
#     "output.LayerNorm",
# ]
# for idx in range(12):
#     for lname in layer_names:
#         wname = f"bert.encoder.layer[{idx}].{lname}.weight"
#         w_train = eval(f"model_train.{wname}")
#         w_eval = eval(f"model_eval.{wname}")
#         print(f"Checking {lname}: {torch.all(w_train==w_eval).item()}")
# wname = f"qa_outputs.weight"
# w_train = eval(f"model_train.{wname}")
# w_eval = eval(f"model_eval.{wname}")
# print(f"Checking qa_outputs.weight: {torch.all(w_train==w_eval).item()}")

compare_models(model_train, model_eval)
