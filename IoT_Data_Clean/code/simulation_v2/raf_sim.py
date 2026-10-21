"""RAF tandem-line simulator, version 2 (independent re-implementation, 2026-10-19).

A 14-station serial line with finite inter-station buffers, stochastic station holds,
asynchronous telemetry (lognormal delay with burst multiplier and packet drop), a finite
c-server authoritative-query service, and stochastic actuation delay. At every station a
release decision moves the next unit from the station's input buffer into service.

Policies
  naive     : release when the latest received observation looks safe, regardless of age
  static    : as naive, but abstain (no query) if the observation is older than 3.0
  fixed     : Fixed + Query; certify if age <= tau, otherwise query
  adaptive  : Adaptive-A; state-dependent tau = 1.5 * clip(1 - 0.5*down_occ + 0.6*min(wait/q_limit,1), .35, 1.5)
  matched   : Matched-State-B; tau = 2.0 * clip(1 - 0.8*down_occ, .35, 1.5)
  oracle    : certify on the true current state (no query); still exposed to actuation races

Execution predicates checked at actuation (true state):
  M (mechanical/resource): station busy or downstream buffer full
  H (permission): station hold active
A failed execution leaves the unit in the buffer; the station retries later.
"""
from __future__ import annotations
import heapq, math
from dataclasses import dataclass, asdict, field
import numpy as np


@dataclass
class Config:
    n_stations: int = 14
    buffer_size: int = 3
    horizon: float = 1200.0
    arrival_mean: float = 8.0
    service_mean: float = 22.0
    service_cv: float = 0.25
    delay_scale: float = 1.0
    delay_sigma: float = 0.85
    burst_prob: float = 0.055
    burst_lo: float = 3.0
    burst_hi: float = 9.0
    packet_drop: float = 0.015
    heartbeat: float = 10.0
    query_servers: int = 2
    query_service_mean: float = 8.0
    actuation_mean: float = 0.35
    hold_mtbf: float = 450.0
    hold_mean: float = 18.0
    stop_mtbf: float = 90.0      # minor stoppages: station unavailable (mechanical), not a permission state
    stop_mean: float = 4.0
    threshold: float = 1.5
    q_limit: float = 220.0
    q0: float = 0.72
    beta: float = 2.8
    retry: float = 1.0
    policy: str = 'adaptive'
    seed: int = 1
    delay_family: str = 'lognormal'      # or 'gamma' (matched mean)
    service_family: str = 'exponential'  # query service; or 'weibull' (shape 1.5, matched mean)
    sample_every: float = 0.0            # >0: record (u, q) samples for reduced-order calibration
    record_events: bool = False          # record hold-involved conflict details


class Sim:
    def __init__(self, c: Config):
        self.c = c
        self.rng = np.random.default_rng(c.seed)
        n = c.n_stations
        self.t = 0.0
        self.ev = []
        self.k = 0
        # true state
        self.buf = [[] for _ in range(n)]          # input buffer of station i (station 0 unbounded)
        self.busy = [False] * n                    # holds a unit (in service or blocked)
        self.blocked = [False] * n
        self.unit_in = [None] * n
        self.hold = [False] * n
        self.stop = [False] * n
        # observations held by the decision service
        self.obs = [dict(src=0.0, busy=False, hold=False, down_occ=0.0, down_full=False) for _ in range(n)]
        self.pending = [False] * n                 # query outstanding or certified action in flight
        self.jobs = {}
        self.nid = 0
        # query service
        self.q_free = c.query_servers
        self.q_wait = []                           # FIFO of (t_req, station)
        self.q_waits = []
        self.q_sojourns = []
        # counters
        self.completed = 0
        self.dispatches = 0
        self.fail = 0
        self.v_hold = 0
        self.v_mech = 0
        self.v_comp = 0
        self.v_race = 0
        self.abstain = 0
        self.queries = 0
        self.blocks = 0
        self.drops = 0
        self.stale = 0
        self.degr = 0.0
        self.degr_grid = np.zeros(11)
        self.q_grid = np.linspace(0.40, 0.90, 11)
        self.waits = []
        self.cycle = []
        self.ages = []
        self.events = []
        self.samples = []

    # ---------------- infrastructure ----------------
    def push(self, t, kind, args=()):
        self.k += 1
        heapq.heappush(self.ev, (t, self.k, kind, args))

    def service_time(self):
        m, cv = self.c.service_mean, self.c.service_cv
        s2 = math.log(1 + cv * cv)
        return float(self.rng.lognormal(math.log(m) - s2 / 2, math.sqrt(s2)))

    def delay(self):
        c = self.c
        if c.delay_family == 'gamma':
            # matched mean of the lognormal base: scale*exp(sigma^2/2), shape chosen to match CV
            mean = c.delay_scale * math.exp(c.delay_sigma ** 2 / 2)
            cv2 = math.exp(c.delay_sigma ** 2) - 1
            k = 1 / cv2
            d = float(self.rng.gamma(k, mean / k))
        else:
            d = float(self.rng.lognormal(math.log(c.delay_scale), c.delay_sigma))
        if self.rng.random() < c.burst_prob:
            d *= float(self.rng.uniform(c.burst_lo, c.burst_hi))
        return d

    def q_service(self):
        m = self.c.query_service_mean
        if self.c.service_family == 'weibull':
            k = 1.5
            return float(m / math.gamma(1 + 1 / k) * self.rng.weibull(k))
        return float(self.rng.exponential(m))

    def true_state(self, i):
        n = self.c.n_stations
        if i == n - 1:
            occ, full = 0.0, False
        else:
            occ = len(self.buf[i + 1]) / self.c.buffer_size
            full = len(self.buf[i + 1]) >= self.c.buffer_size
        return dict(busy=self.busy[i] or self.stop[i], hold=self.hold[i], down_occ=occ, down_full=full)

    def publish(self, i):
        """Station i emits a status message; delivered later or dropped."""
        if self.rng.random() < self.c.packet_drop:
            self.drops += 1
            return
        st = self.true_state(i)
        st['src'] = self.t
        self.push(self.t + self.delay(), 'deliver', (i, st))

    def publish_neigh(self, i):
        self.publish(i)
        if i > 0:
            self.publish(i - 1)   # upstream station's downstream occupancy changed

    # ---------------- policy ----------------
    def safe(self, o):
        return (not o['busy']) and (not o['hold']) and (not o['down_full'])

    def tau(self, i, j):
        c = self.c
        o = self.obs[i]
        if c.policy == 'adaptive':
            qage = max(0.0, self.t - self.jobs[j]) / c.q_limit
            f = min(1.5, max(0.35, 1.0 - 0.5 * o['down_occ'] + 0.6 * min(qage, 1.0)))
            return 1.5 * f
        if c.policy == 'matched':
            f = min(1.5, max(0.35, 1.0 - 0.8 * o['down_occ']))
            return 2.0 * f
        return c.threshold

    def attempt(self, i):
        c = self.c
        if self.pending[i] or not self.buf[i] or self.t >= c.horizon:
            return
        j = self.buf[i][0]
        if c.policy == 'oracle':
            st = self.true_state(i)
            if self.safe(st):
                self.certify(i, j, 0.0, st['hold'])
            else:
                self.abstain += 1
                self.push(self.t + c.retry, 'retry', (i,))
            return
        o = self.obs[i]
        age = max(0.0, self.t - o['src'])
        if not self.safe(o):
            self.abstain += 1
            self.push(self.t + c.retry, 'retry', (i,))
            return
        if c.policy == 'naive':
            self.certify(i, j, age, o['hold']); return
        if c.policy == 'static':
            if age <= 3.0:
                self.certify(i, j, age, o['hold'])
            else:
                self.abstain += 1
                self.push(self.t + c.retry, 'retry', (i,))
            return
        if age <= self.tau(i, j):
            self.certify(i, j, age, o['hold'])
        else:
            self.pending[i] = True
            self.queries += 1
            self.q_wait.append((self.t, i))
            self.serve_queries()

    def serve_queries(self):
        while self.q_free > 0 and self.q_wait:
            tr, i = self.q_wait.pop(0)
            self.q_free -= 1
            self.q_waits.append(self.t - tr)
            self.push(self.t + self.q_service(), 'qdone', (i, tr))

    def certify(self, i, j, age, hold_seen):
        self.pending[i] = True
        self.ages.append(age)
        a = float(self.rng.exponential(self.c.actuation_mean)) if self.c.actuation_mean > 0 else 0.0
        self.push(self.t + a, 'execute', (i, j, self.t, self.hold[i]))

    # ---------------- events ----------------
    def run(self):
        c = self.c
        self.push(float(self.rng.exponential(c.arrival_mean)), 'arrive')
        for i in range(c.n_stations):
            self.push(float(self.rng.exponential(c.hold_mtbf)), 'hold_on', (i,))
            self.push(float(self.rng.uniform(0, c.heartbeat)), 'heartbeat', (i,))
            if c.stop_mtbf > 0:
                self.push(float(self.rng.exponential(c.stop_mtbf)), 'stop_on', (i,))
            self.publish(i)
        if c.sample_every > 0:
            self.push(c.sample_every, 'sample')
        while self.ev:
            t, _, kind, args = heapq.heappop(self.ev)
            if t > c.horizon:
                break
            self.t = t
            getattr(self, 'on_' + kind)(*args)
        return self.result()

    def on_arrive(self):
        j = self.nid; self.nid += 1
        self.jobs[j] = self.t
        self.jobs_born = getattr(self, 'jobs_born', {}); self.jobs_born[j] = self.t
        self.buf[0].append(j)
        self.push(self.t + float(self.rng.exponential(self.c.arrival_mean)), 'arrive')
        self.attempt(0)

    def on_retry(self, i):
        self.attempt(i)

    def on_heartbeat(self, i):
        self.publish(i)
        self.push(self.t + self.c.heartbeat, 'heartbeat', (i,))

    def on_deliver(self, i, st):
        if st['src'] < self.obs[i]['src']:
            self.stale += 1
            return
        self.obs[i] = st
        if not self.pending[i]:
            self.attempt(i)

    def on_qdone(self, i, tr):
        self.q_free += 1
        self.q_sojourns.append(self.t - tr)
        st = self.true_state(i); st['src'] = self.t
        self.obs[i] = st
        self.pending[i] = False
        self.serve_queries()
        if self.buf[i] and self.safe(st):
            self.certify(i, self.buf[i][0], 0.0, st['hold'])
        else:
            self.attempt(i)

    def on_hold_on(self, i):
        self.hold[i] = True
        self.publish(i)
        self.push(self.t + float(self.rng.exponential(self.c.hold_mean)), 'hold_off', (i,))

    def on_stop_on(self, i):
        self.stop[i] = True
        self.publish(i)
        self.push(self.t + float(self.rng.exponential(self.c.stop_mean)), 'stop_off', (i,))

    def on_stop_off(self, i):
        self.stop[i] = False
        self.publish(i)
        self.push(self.t + float(self.rng.exponential(self.c.stop_mtbf)), 'stop_on', (i,))
        self.attempt(i)

    def on_hold_off(self, i):
        self.hold[i] = False
        self.publish(i)
        self.push(self.t + float(self.rng.exponential(self.c.hold_mtbf)), 'hold_on', (i,))
        self.attempt(i)

    def on_execute(self, i, j, t_cert, hold_at_cert):
        self.pending[i] = False
        if not self.buf[i] or self.buf[i][0] != j:
            self.attempt(i); return
        self.dispatches += 1
        st = self.true_state(i)
        M = st['busy'] or st['down_full']
        H = st['hold']
        if M or H:
            self.fail += 1
            if M and H: self.v_comp += 1
            elif H: self.v_hold += 1
            else: self.v_mech += 1
            if H and not hold_at_cert:
                self.v_race += 1
            if H and self.c.record_events:
                self.events.append(dict(seed=self.c.seed, t_cert=t_cert, t_exec=self.t,
                                        hold_cert=hold_at_cert, hold_exec=True,
                                        busy_exec=st['busy'], down_full_exec=st['down_full']))
            self.push(self.t + self.c.retry, 'retry', (i,))
            return
        # successful release
        self.buf[i].pop(0)
        w = self.t - self.jobs[j]
        self.waits.append(w)
        q = w / self.c.q_limit
        self.degr += self.c.beta * max(q - self.c.q0, 0.0) ** 2
        self.degr_grid += np.maximum(q - self.q_grid, 0.0) ** 2
        self.busy[i] = True
        self.unit_in[i] = j
        # the decision service knows its own committed release: local ledger marks the station busy
        o = dict(self.obs[i]); o['busy'] = True; o['src'] = self.t
        self.obs[i] = o
        self.publish_neigh(i)
        self.push(self.t + self.service_time(), 'done', (i,))
        if i > 0:
            self.try_unblock(i - 1)

    def on_done(self, i):
        self.move_out(i)

    def move_out(self, i):
        n = self.c.n_stations
        j = self.unit_in[i]
        if i == n - 1:
            self.completed += 1
            self.cycle.append(self.t - self.jobs_born[j])
        else:
            if len(self.buf[i + 1]) >= self.c.buffer_size:
                if not self.blocked[i]:
                    self.blocked[i] = True
                    self.blocks += 1
                return
            self.buf[i + 1].append(j)
            self.jobs[j] = self.t
            self.attempt(i + 1)
        self.blocked[i] = False
        self.busy[i] = False
        self.unit_in[i] = None
        self.publish_neigh(i)
        self.attempt(i)
        if i > 0:
            self.try_unblock(i - 1)

    def try_unblock(self, i):
        if self.blocked[i]:
            self.move_out(i)

    def on_sample(self):
        c = self.c
        ex = []
        for i in range(c.n_stations):
            if self.buf[i] and not self.pending[i] and self.safe(self.obs[i]):
                age = self.t - self.obs[i]['src']
                ex.append(max(0.0, age - self.tau(i, self.buf[i][0])))
        u = float(np.mean(ex)) if ex else 0.0
        outstanding = len(self.q_wait) + (c.query_servers - self.q_free)
        self.samples.append((self.t, u, outstanding / c.query_servers))
        self.push(self.t + c.sample_every, 'sample')

    # ---------------- outputs ----------------
    def result(self):
        c = self.c
        d = self.dispatches
        H = self.v_hold + self.v_comp
        r = asdict(c)
        r.pop('record_events'); r.pop('sample_every')
        lam = self.queries / c.horizon
        r.update(
            completed=self.completed, dispatches=d, failed=self.fail,
            mech_only=self.v_mech, hold_only=self.v_hold, compound=self.v_comp,
            hold_involved=H, race=self.v_race,
            aggregate_conflict=self.fail / d if d else 0.0,
            hold_rate=H / d if d else 0.0,
            abstentions=self.abstain, queries=self.queries,
            query_rate=self.queries / (d + self.queries) if (d + self.queries) else 0.0,
            query_wait_mean=float(np.mean(self.q_waits)) if self.q_waits else 0.0,
            query_wait_p95=float(np.percentile(self.q_waits, 95)) if self.q_waits else 0.0,
            query_sojourn_p95=float(np.percentile(self.q_sojourns, 95)) if self.q_sojourns else 0.0,
            offered_load=lam * c.query_service_mean / c.query_servers,
            block_events=self.blocks, telemetry_drops=self.drops, stale_discards=self.stale,
            mean_wait=float(np.mean(self.waits)) if self.waits else 0.0,
            p95_cycle=float(np.percentile(self.cycle, 95)) if self.cycle else 0.0,
            action_age_p95=float(np.percentile(self.ages, 95)) if self.ages else 0.0,
            degradation=self.degr,
        )
        for q, v in zip(self.q_grid, self.degr_grid):
            r[f'degr_unit_q0_{int(round(q*100))}'] = float(v)
        return r


def run(c: Config):
    s = Sim(c)
    r = s.run()
    return r, s
