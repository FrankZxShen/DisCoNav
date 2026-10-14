python main_tpami_250903.py -d ./VLM_EXP/multi_hm3d_2-robot_tpami_lambda0.5_250903/  \
                    --num_agents 2 --visualize 2 \
                    --consensus_lambda 0.5 --voting_alpha 0.7 --auction_beta 0.8 \
                    --num_sem_categories 16 \
                    --task_config tasks/multi_objectnav_hm3d.yaml \
                    --base_url http://127.0.0.1:31516
