from utils.my_utils import load_json, dump_json
from models.llms_case_study import (ChatGPTTLCS_Domain_Knowledge, ChatGPTTLCS_Commonsense_No_Calculation, ChatGPTTLCS_Commonsense_Flow_Coordination, ChatGPTTLCS_Commonsense_Zero,
                                    LLM_TLCS_Domain_Knowledge, LLM_TLCS_Commonsense_No_Calculation, LLM_TLCS_Commonsense_Flow_Coordination, LLM_TLCS_Commonsense_No_Zero)
from tqdm import tqdm
# case_path = "/data/xuzhao/CaseStudy/ChatGPT4TSCS/logs_case_study/anon_3_4_jinan_real.json11_12_10_41_46/case_study/"
case_path = "./logs_case_study/anon_3_4_jinan_real.json11_12_10_41_46/case_study/"

case_filename = "5_input.json"

# case_path = "./logs_case_study/flow_3_2_3_legs.json11_17_14_44_41/case_study/"

# case_filename = "1_input.json"

cases = load_json(case_path + case_filename)

road_lengths = {"North": 800.0, "South": 800.0, "East": 400.0, "West": 400.0} # jinan
# road_lengths = {"North": 300.0, "South": 300.0, "East": 300.0, "West": 300.0} # 3 leg

# GPT_domain = ChatGPTTLCS_Domain_Knowledge('gpt-4', 4, road_lengths)
# GPT_common = ChatGPTTLCS_Commonsense_No_Calculation('gpt-4', 4, road_lengths)
# GPT_coordination = ChatGPTTLCS_Commonsense_Flow_Coordination('gpt-4', 4, road_lengths)
# llm_domain = LLM_TLCS_Domain_Knowledge('llama_2_70b_chat_hf', 4, road_lengths)
# llm_common = LLM_TLCS_Commonsense_No_Calculation('llama_2_70b_chat_hf', 4, road_lengths)
# llm_coordination = LLM_TLCS_Commonsense_Flow_Coordination('llama_2_70b_chat_hf', 4, road_lengths)

GPT_zero = ChatGPTTLCS_Commonsense_Zero('gpt-4', 4, road_lengths)
llm_zero = LLM_TLCS_Commonsense_No_Zero('llama_2_70b_chat_hf', 4, road_lengths)

state_action_logs = []

for state in tqdm(cases):
    # GPT_domain_res = GPT_domain.choose_action(state)
    # tqdm.write("Finish GPT_domain")
    # GPT_common_res = GPT_common.choose_action(state)
    # tqdm.write("Finish GPT_common")
    # GPT_coordination_res = GPT_coordination.choose_action(state)
    # tqdm.write("Finish GPT_coordination")
    # llm_domain_res = llm_domain.choose_action(state)
    # tqdm.write("Finish llm_domain")
    # llm_common_res = llm_common.choose_action(state)
    # tqdm.write("Finish llm_common")
    # llm_coordination_res = llm_coordination.choose_action(state)
    # tqdm.write("Finish llm_coordination")

    GPT_zero_res = GPT_zero.choose_action(state)
    tqdm.write("Finish GPT_zero")
    llm_zero_res = llm_zero.choose_action(state)
    tqdm.write("Finish llm_zero")

    state_action_logs.append({"state": state, "models": {
                                                         "GPT_zero": GPT_zero_res,
                                                         "llm_zero": llm_zero_res
                                                         # "GPT_domain": GPT_domain_res,
                                                         # "GPT_common": GPT_common_res,
                                                         # "GPT_coordination": GPT_coordination_res,
                                                         # "llm_domain": llm_domain_res,
                                                         # "llm_common": llm_common_res,
                                                         # "llm_coordination": llm_coordination_res
                                                         }})

dump_json(state_action_logs, case_path + "llm_cases_state_action_zero_logs.json")



