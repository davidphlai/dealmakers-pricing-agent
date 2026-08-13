import pickle
import numpy as np


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.remaining_inventory = params['inventory_limit']

        # Segment thresholds from the training data
        self.t1 = 2.7193025761078644
        self.t2 = 2.7215555543935457
        self.t3 = 7.262601783583493

        # 8 segment demand models
        with open('agents/dealmakers/8_models_dict.pkl', 'rb') as f:
            self.models = pickle.load(f)

        # Precomputed 8 segment DP pricing policy
        with open('agents/dealmakers/dp_policy.pkl', 'rb') as f:
            self.dp_policy = pickle.load(f)

        self.seg_multipliers = {key: 1.0 for key in self.dp_policy.keys()}
        self.seg_sale_history = {key: [] for key in self.dp_policy.keys()}

        self.last_seg_key = None
        self.last_price = 100.0

        self.opponent_price_history = []
        self.my_price_history = []
        self.last_sale_winner = None

        self.PRICE_GRID = np.linspace(0.01, 500, 100)

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):

        self.remaining_inventory = inventories[self.this_agent_number]

        if last_sale[0] is None:
            return

        winner = last_sale[0]
        self.last_sale_winner = winner

        my_price = last_sale[1][self.this_agent_number]
        opp_price = last_sale[1][1 - self.this_agent_number]

        self.my_price_history.append(my_price)
        self.opponent_price_history.append(opp_price)

        if len(self.my_price_history) > 10:
            self.my_price_history.pop(0)
            self.opponent_price_history.pop(0)

        if self.last_seg_key is None:
            return

        seg_key = self.last_seg_key
        history = self.seg_sale_history[seg_key]

        did_buy = (winner == self.this_agent_number)
        history.append(1 if did_buy else 0)
        if len(history) > 5:
            history.pop(0)

        if len(history) >= 3:
            br = sum(history) / len(history)
            m = self.seg_multipliers[seg_key]

            if br >= 0.8:
                m *= 1.15
            elif br <= 0.2:
                m *= 0.90
            elif br >= 0.6:
                m *= 1.05
            elif br <= 0.4:
                m *= 0.97

            self.seg_multipliers[seg_key] = np.clip(m, 0.8, 1.3)

    def action(self, obs):

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        C1, C2, C3 = new_buyer_covariates
        seg_key = (C1 > self.t1, C2 > self.t2, C3 > self.t3)
        self.last_seg_key = seg_key

        I = int(self.remaining_inventory)
        if I <= 0:
            return 999.0

        t = max(0, min(time_until_replenish, self.dp_policy[seg_key].shape[1] - 1))
        p_dp = self.dp_policy[seg_key][I][t]
        if p_dp <= 0:
            p_dp = 50.0

        m = self.seg_multipliers[seg_key]
        p_dp = p_dp * m

        best_p = 100
        best_rev = -1
        model = self.models[seg_key]

        for p_test in self.PRICE_GRID:
            rev = p_test * model.predict_proba([[C1, C2, C3, p_test]])[0, 1]
            if rev > best_rev:
                best_rev = rev
                best_p = p_test

        p_static = best_p * m

        prob_dp = model.predict_proba([[C1, C2, C3, p_dp]])[0, 1]
        prob_static = model.predict_proba([[C1, C2, C3, p_static]])[0, 1]

        rev_dp = p_dp * prob_dp
        rev_static = p_static * prob_static

        if rev_static > rev_dp * 1.03:
            p_final = p_static
        else:
            p_final = p_dp

        opp_last = last_sale[1][1 - self.this_agent_number]

        if opp_last > 0:

            if p_final >= opp_last:
                p_final = opp_last - 0.5

            if len(self.opponent_price_history) >= 3:
                if (self.opponent_price_history[-1] < self.my_price_history[-1] and
                    self.opponent_price_history[-2] < self.my_price_history[-2] and
                    self.opponent_price_history[-3] < self.my_price_history[-3]):

                    p_final = min(p_final, opp_last - 2.0)


        p_final = float(np.clip(p_final, 5.0, 500.0))
        self.last_price = p_final

        return p_final

