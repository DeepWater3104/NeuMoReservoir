prompts = [i for i in range(10)]
prompt_codes = [f"{i:02d}" for i in prompts]
train_session_codes = ["se"]
test_session_codes  = ["s2", "s3", "s4", "s5", "s6", "s7", "s8"]
train_token_codes    = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
test_token_codes   = ["t0", "t1"]
speaker_codes = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]

# all data spoken by "f1"
trainseq_code = {} 
trainseq_code['trainseq_promptcode']  = [prompt for prompt in prompts
                                            for i in range(len(train_session_codes))
                                            for j in range(len(train_token_codes))]
trainseq_code['trainseq_speakercode'] = ["f1" for i in range(len(prompt_codes)) 
                                            for j in range(len(train_session_codes))
                                            for k in range(len(train_token_codes))]
trainseq_code['trainseq_sessioncode'] = [session for i in range(len(prompt_codes)) 
                                               for session in train_session_codes
                                               for j in range(len(train_token_codes))]
trainseq_code['trainseq_tokencode']   = [token for i in range(len(prompt_codes)) 
                                               for j in range(len(train_session_codes))
                                               for token in train_token_codes]
testseq_code = {} 
testseq_code['testseq_promptcode']  = [prompt for prompt in prompts
                                            for i in range(len(test_session_codes))
                                            for j in range(len(test_token_codes))]
testseq_code['testseq_speakercode'] = ["f1" for i in range(len(prompt_codes)) 
                                            for j in range(len(test_session_codes))
                                            for k in range(len(test_token_codes))]
testseq_code['testseq_sessioncode'] = [session for i in range(len(prompt_codes)) 
                                               for session in test_session_codes
                                               for j in range(len(test_token_codes))]
testseq_code['testseq_tokencode']   = [token for i in range(len(prompt_codes)) 
                                               for j in range(len(test_session_codes))
                                               for token in test_token_codes]

# all zero and one data spoken by "f1" 
#trainseq_code = {} 
#trainseq_code['trainseq_promptcode']  = [0 for i in range(len(train_session_codes))
#                                           for j in range(len(train_token_codes))]
#trainseq_code['trainseq_speakercode'] = ["f1" for i in range(len(train_session_codes))
#                                              for j in range(len(train_token_codes))]
#trainseq_code['trainseq_sessioncode'] = [session for session in train_session_codes
#                                                 for j in range(len(train_token_codes))]
#trainseq_code['trainseq_tokencode']   = [token for i in range(len(train_session_codes))
#                                               for token in train_token_codes]
#trainseq_code['trainseq_promptcode'].extend([1 for i in range(len(train_session_codes))
#                                        for j in range(len(train_token_codes))])
#trainseq_code['trainseq_speakercode'].extend(["f1" for i in range(len(train_session_codes))
#                                           for j in range(len(train_token_codes))])
#trainseq_code['trainseq_sessioncode'].extend([session for session in train_session_codes
#                                                 for j in range(len(train_token_codes))])
#trainseq_code['trainseq_tokencode'].extend([token for i in range(len(train_session_codes))
#                                               for token in train_token_codes])
#
#testseq_code = {} 
#testseq_code['testseq_promptcode']  = [0 for i in range(len(test_session_codes))
#                                           for j in range(len(test_token_codes))]
#testseq_code['testseq_speakercode'] = ["f1" for i in range(len(test_session_codes))
#                                              for j in range(len(test_token_codes))]
#testseq_code['testseq_sessioncode'] = [session for session in test_session_codes
#                                                 for j in range(len(test_token_codes))]
#testseq_code['testseq_tokencode']   = [token for i in range(len(test_session_codes))
#                                                 for token in test_token_codes]
#testseq_code['testseq_promptcode'].extend([1 for i in range(len(test_session_codes))
#                                        for j in range(len(test_token_codes))])
#testseq_code['testseq_speakercode'].extend(["f1" for i in range(len(test_session_codes))
#                                           for j in range(len(test_token_codes))])
#testseq_code['testseq_sessioncode'].extend([session for session in test_session_codes
#                                              for j in range(len(test_token_codes))])
#testseq_code['testseq_tokencode'].extend([token for i in range(len(test_session_codes))
#                                                 for token in test_token_codes])

# single zero and one data spoken by "f1"
#trainseq_code = {} 
#trainseq_code['trainseq_promptcode']  = [0, 1]
#trainseq_code['trainseq_speakercode'] = ["f1", "f1"]
#trainseq_code['trainseq_sessioncode'] = [train_session_codes[0], train_session_codes[0]]
#trainseq_code['trainseq_tokencode']   = [train_token_codes[0], train_token_codes[0]]
#testseq_code = {} 
#testseq_code['testseq_promptcode']  = [0, 1]
#testseq_code['testseq_speakercode'] = ["f1", "f1"]
#testseq_code['testseq_sessioncode'] = [test_session_codes[0], test_session_codes[0]]
#testseq_code['testseq_tokencode']   = [test_token_codes[0], test_token_codes[0]]

# single one spoken by "f1"
#trainseq_code = {} 
#trainseq_code['trainseq_promptcode']  = [0]
#trainseq_code['trainseq_speakercode'] = ["f1"]
#trainseq_code['trainseq_sessioncode'] = [train_session_codes[0]]
#trainseq_code['trainseq_tokencode']   = [train_token_codes[0]]
#testseq_code = {} 
#testseq_code['testseq_promptcode']  = [0]
#testseq_code['testseq_speakercode'] = ["f1"]
#testseq_code['testseq_sessioncode'] = [test_session_codes[0]]
#testseq_code['testseq_tokencode']   = [test_token_codes[0]]
