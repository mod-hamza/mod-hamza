#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub stats generator (user + orgs).

- User data via user(login: ...)
- Org data via organization(login: ...)
- Auto-detect which PAT is the user token; map the rest to ORG_LOGINS
- Aggregates repos, stars, LOC, and commits across user + orgs
- Preserves original printing format
"""

from __future__ import annotations

from dateutil import relativedelta
from xml.dom import minidom
from pathlib import Path
import datetime
import requests
import hashlib
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Iterable

# -------------------------
# Environment
# -------------------------

USER_NAME = os.environ.get('USER_LOGIN', 'mod-hamza').strip()

TOKENS_STR = os.environ.get('ACCESS_TOKEN', '').strip()
if not TOKENS_STR:
    # .env fallback
    env_path = Path('.env')
    if env_path.exists():
        for raw in env_path.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            if k.strip() == 'ACCESS_TOKEN':
                TOKENS_STR = v.strip().strip("\"").strip('\'')
                break
if not TOKENS_STR:
    raise SystemExit("ACCESS_TOKEN not set. Provide comma-separated PATs in env or .env")

TOKENS: List[str] = [t.strip() for t in TOKENS_STR.split(',') if t.strip()]
if not TOKENS:
    raise SystemExit("No valid tokens in ACCESS_TOKEN")

ORG_LOGINS = [s.strip() for s in os.environ.get('ORG_LOGINS', '').split(',') if s.strip()]

HEADERS_LIST = [{'authorization': f'Bearer {t}'} for t in TOKENS]

GQL_URL = 'https://api.github.com/graphql'
SESSION = requests.Session()
SESSION.headers.update({'Accept': 'application/vnd.github+json'})

QUERY_COUNT = {
    'user_getter': 0,
    'follower_getter': 0,
    'graph_repos_stars_user': 0,
    'graph_repos_stars_org': 0,
    'recursive_loc': 0,
    'graph_commits': 0,
    'loc_query_user': 0,
    'loc_query_org': 0,
    'loc_query_total': 0,
    'graphql': 0,
}

# -------------------------
# GraphQL helpers
# -------------------------

def _post(token_idx: int, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    QUERY_COUNT['graphql'] += 1
    r = SESSION.post(GQL_URL, json={'query': query, 'variables': variables}, headers=HEADERS_LIST[token_idx], timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GraphQL HTTP {r.status_code}: {r.text}")
    j = r.json()
    if 'errors' in j and j['errors']:
        raise RuntimeError(f"GraphQL errors: {j['errors']}")
    return j['data']

Q_VIEWER = "query { viewer { login } }"
Q_ORG_PING = 'query($login:String!){ organization(login:$login){ login } }'

def _viewer_login(token_idx: int) -> Optional[str]:
    try:
        data = _post(token_idx, Q_VIEWER, {})
        return data['viewer']['login']
    except Exception:
        return None

def _can_see_org(token_idx: int, org: str) -> bool:
    try:
        data = _post(token_idx, Q_ORG_PING, {'login': org})
        return bool(data['organization'] and data['organization']['login'] == org)
    except Exception:
        return False

def auto_map_tokens(user_login: str, orgs: List[str]) -> Tuple[int, List[int]]:
    # pick token whose viewer == USER_NAME
    user_idx = None
    for i in range(len(HEADERS_LIST)):
        v = _viewer_login(i)
        if v and v.lower() == user_login.lower():
            user_idx = i
            break
    if user_idx is None:
        raise SystemExit("No token authenticates as USER_LOGIN. Grant SSO/scopes or rotate PATs.")
    used = {user_idx}
    org_indices: List[int] = []
    for org in orgs:
        found = None
        for i in range(len(HEADERS_LIST)):
            if i in used:
                continue
            if _can_see_org(i, org):
                found = i
                break
        if found is None:
            raise SystemExit(f"No token can access org '{org}'. Grant SSO and repo permissions to one PAT.")
        org_indices.append(found)
        used.add(found)
    return user_idx, org_indices

# -------------------------
# Original utilities
# -------------------------

def daily_readme(birthday):
    diff = relativedelta.relativedelta(datetime.datetime.today(), birthday)
    return '{} {}, {} {}, {} {}{}'.format(
        diff.years, 'year' + format_plural(diff.years),
        diff.months, 'month' + format_plural(diff.months),
        diff.days, 'day' + format_plural(diff.days),
        ' 🎂' if (diff.months == 0 and diff.days == 0) else '')

def format_plural(unit):
    return 's' if unit != 1 else ''

def perf_counter(funct, *args):
    start = time.perf_counter()
    ret = funct(*args)
    return ret, time.perf_counter() - start

def formatter(query_type, difference, funct_return=False, whitespace=0):
    if whitespace:
        return f"{funct_return:,}"
    return funct_return

# -------------------------
# Queries: user
# -------------------------

def simple_request_user(func_name, query, variables, token_idx: int):
    request = SESSION.post(GQL_URL, json={'query': query, 'variables': variables}, headers=HEADERS_LIST[token_idx], timeout=60)
    if request.status_code == 200:
        return request
    raise Exception(func_name, ' has failed with a', request.status_code, request.text, QUERY_COUNT)

def user_getter(username, token_idx: int):
    QUERY_COUNT['user_getter'] += 1
    query = '''
    query($login: String!){
        user(login: $login) {
            id
            createdAt
        }
    }'''
    variables = {'login': username}
    request = simple_request_user(user_getter.__name__, query, variables, token_idx)
    return {'id': request.json()['data']['user']['id']}, request.json()['data']['user']['createdAt']

def follower_getter(username, token_idx: int):
    QUERY_COUNT['follower_getter'] += 1
    query = '''
    query($login: String!){
        user(login: $login) {
            followers {
                totalCount
            }
        }
    }'''
    request = simple_request_user(follower_getter.__name__, query, {'login': username}, token_idx)
    return int(request.json()['data']['user']['followers']['totalCount'])

def graph_commits(start_date, end_date, token_idx: int):
    QUERY_COUNT['graph_commits'] += 1
    query = '''
    query($start_date: DateTime!, $end_date: DateTime!, $login: String!) {
        user(login: $login) {
            contributionsCollection(from: $start_date, to: $end_date) {
                contributionCalendar {
                    totalContributions
                }
            }
        }
    }'''
    variables = {'start_date': start_date,'end_date': end_date, 'login': USER_NAME}
    request = simple_request_user(graph_commits.__name__, query, variables, token_idx)
    return int(request.json()['data']['user']['contributionsCollection']['contributionCalendar']['totalContributions'])

def graph_repos_stars_user(count_type, owner_affiliation, token_idx: int, cursor=None):
    QUERY_COUNT['graph_repos_stars_user'] += 1
    query = '''
    query ($owner_affiliation: [RepositoryAffiliation], $login: String!, $cursor: String) {
        user(login: $login) {
            repositories(first: 100, after: $cursor, ownerAffiliations: $owner_affiliation) {
                totalCount
                edges {
                    node {
                        ... on Repository {
                            nameWithOwner
                            stargazers { totalCount }
                        }
                    }
                }
                pageInfo { endCursor hasNextPage }
            }
        }
    }'''
    variables = {'owner_affiliation': owner_affiliation, 'login': USER_NAME, 'cursor': cursor}
    r = simple_request_user(graph_repos_stars_user.__name__, query, variables, token_idx)
    data = r.json()['data']['user']['repositories']
    if count_type == 'repos':
        return data['totalCount']
    elif count_type == 'stars':
        return stars_counter(data['edges'])
    return 0

# -------------------------
# Queries: orgs
# -------------------------

def graph_repos_stars_org(count_type, org_login: str, token_idx: int, cursor=None):
    QUERY_COUNT['graph_repos_stars_org'] += 1
    query = '''
    query ($login: String!, $cursor: String) {
      organization(login: $login) {
        repositories(first: 100, after: $cursor, orderBy: {field: NAME, direction: ASC}) {
          totalCount
          edges {
            node {
              ... on Repository {
                nameWithOwner
                stargazers { totalCount }
              }
            }
          }
          pageInfo { endCursor hasNextPage }
        }
      }
    }'''
    variables = {'login': org_login, 'cursor': cursor}
    r = _post(token_idx, query, variables)
    repos = r['organization']['repositories']
    if count_type == 'repos':
        return repos['totalCount']
    elif count_type == 'stars':
        return stars_counter(repos['edges'])
    return 0

def list_repos_user(token_idx: int, owner_affiliation: List[str]) -> List[Dict[str, Any]]:
    QUERY_COUNT['loc_query_user'] += 1
    query = '''
    query ($owner_affiliation: [RepositoryAffiliation], $login: String!, $cursor: String) {
        user(login: $login) {
            repositories(first: 60, after: $cursor, ownerAffiliations: $owner_affiliation) {
                edges {
                    node {
                        ... on Repository {
                            nameWithOwner
                            isPrivate
                            defaultBranchRef {
                                target {
                                    ... on Commit { history { totalCount } }
                                }
                            }
                        }
                    }
                }
                pageInfo { endCursor hasNextPage }
            }
        }
    }'''
    edges: List[Dict[str, Any]] = []
    cursor = None
    while True:
        r = _post(token_idx, query, {'owner_affiliation': owner_affiliation, 'login': USER_NAME, 'cursor': cursor})
        page = r['user']['repositories']
        edges.extend(page['edges'])
        if not page['pageInfo']['hasNextPage']:
            break
        cursor = page['pageInfo']['endCursor']
    return edges

def list_repos_org(token_idx: int, org_login: str) -> List[Dict[str, Any]]:
    QUERY_COUNT['loc_query_org'] += 1
    query = '''
    query ($login: String!, $cursor: String) {
      organization(login: $login) {
        repositories(first: 60, after: $cursor, orderBy: {field: NAME, direction: ASC}) {
          edges {
            node {
              ... on Repository {
                nameWithOwner
                isPrivate
                defaultBranchRef {
                  target {
                    ... on Commit { history { totalCount } }
                  }
                }
              }
            }
          }
          pageInfo { endCursor hasNextPage }
        }
      }
    }'''
    edges: List[Dict[str, Any]] = []
    cursor = None
    while True:
        r = _post(token_idx, query, {'login': org_login, 'cursor': cursor})
        page = r['organization']['repositories']
        edges.extend(page['edges'])
        if not page['pageInfo']['hasNextPage']:
            break
        cursor = page['pageInfo']['endCursor']
    return edges

# -------------------------
# LOC computation (shared)
# -------------------------

def recursive_loc(token_idx: int, owner: str, repo_name: str, data_lines: List[str], cache_comment: List[str],
                  addition_total=0, deletion_total=0, my_commits=0, cursor=None):
    QUERY_COUNT['recursive_loc'] += 1
    query = '''
    query ($repo_name: String!, $owner: String!, $cursor: String) {
        repository(name: $repo_name, owner: $owner) {
            defaultBranchRef {
                target {
                    ... on Commit {
                        history(first: 100, after: $cursor) {
                            totalCount
                            edges {
                                node {
                                    ... on Commit { committedDate }
                                    author { user { id login } }
                                    deletions
                                    additions
                                }
                            }
                            pageInfo { endCursor hasNextPage }
                        }
                    }
                }
            }
        }
    }'''
    variables = {'repo_name': repo_name, 'owner': owner, 'cursor': cursor}
    request = SESSION.post(GQL_URL, json={'query': query, 'variables': variables}, headers=HEADERS_LIST[token_idx], timeout=60)
    if request.status_code == 200:
        repo = request.json()['data']['repository']
        if repo and repo['defaultBranchRef'] is not None:
            return loc_counter_one_repo(token_idx, owner, repo_name, data_lines, cache_comment,
                                        repo['defaultBranchRef']['target']['history'],
                                        addition_total, deletion_total, my_commits)
        else:
            return 0
    force_close_file(data_lines, cache_comment)
    if request.status_code == 403:
        raise Exception('Too many requests. You hit an anti-abuse limit.')
    raise Exception('recursive_loc() failed with', request.status_code, request.text, QUERY_COUNT)

def loc_counter_one_repo(token_idx: int, owner: str, repo_name: str, data_lines: List[str],
                         cache_comment: List[str], history: Dict[str, Any],
                         addition_total: int, deletion_total: int, my_commits: int):
    for node in history['edges']:
        my_commits += 1
        addition_total += node['node']['additions']
        deletion_total += node['node']['deletions']
    if not history['edges'] or not history['pageInfo']['hasNextPage']:
        return addition_total, deletion_total, my_commits
    else:
        return recursive_loc(token_idx, owner, repo_name, data_lines, cache_comment,
                             addition_total, deletion_total, my_commits, history['pageInfo']['endCursor'])

def cache_file_name() -> str:
    key = USER_NAME + '|' + '|'.join(ORG_LOGINS)
    return 'cache/' + hashlib.sha256(key.encode('utf-8')).hexdigest() + '.txt'

def cache_builder(edges: List[Dict[str, Any]], comment_size: int, force_cache: bool,
                  token_for_repo: Dict[str, int], loc_add=0, loc_del=0):
    cached = True
    filename = cache_file_name()
    try:
        with open(filename, 'r') as f:
            data = f.readlines()
    except FileNotFoundError:
        data = []
        if comment_size > 0:
            for _ in range(comment_size):
                data.append('This line is a comment block. Write whatever you want here.\n')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            f.writelines(data)

    if len(data) - comment_size != len(edges) or force_cache:
        cached = False
        flush_cache(edges, filename, comment_size)
        with open(filename, 'r') as f:
            data = f.readlines()

    cache_comment = data[:comment_size]
    data = data[comment_size:]
    for index in range(len(edges)):
        nwo = edges[index]['node']['nameWithOwner']
        repo_hash = hashlib.sha256(nwo.encode('utf-8')).hexdigest()
        if index >= len(data) or len(data[index].split()) < 5:
            data_line = f"{repo_hash} 0 0 0 0\n"
            if index < len(data):
                data[index] = data_line
            else:
                data.append(data_line)
        curr_hash, commit_count, *__ = data[index].split()
        if curr_hash == repo_hash:
            hist = edges[index]['node']['defaultBranchRef']
            try:
                new_total = hist['target']['history']['totalCount'] if hist else 0
                if int(commit_count) != new_total:
                    owner, repo_name = nwo.split('/')
                    token_idx = token_for_repo[nwo]
                    loc = recursive_loc(token_idx, owner, repo_name, data, cache_comment)
                    data[index] = f"{repo_hash} {new_total} {loc[2]} {loc[0]} {loc[1]}\n"
            except TypeError:
                data[index] = f"{repo_hash} 0 0 0 0\n"

    with open(filename, 'w') as f:
        f.writelines(cache_comment)
        f.writelines(data)

    for line in data:
        parts = line.split()
        if len(parts) >= 5:
            loc_add += int(parts[3])
            loc_del += int(parts[4])
    return [loc_add, loc_del, loc_add - loc_del, cached]

def flush_cache(edges: List[Dict[str, Any]], filename: str, comment_size: int):
    try:
        with open(filename, 'r') as f:
            data = []
            if comment_size > 0:
                data = f.readlines()[:comment_size]
    except FileNotFoundError:
        data = []
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(data)
        for node in edges:
            nwo = node['node']['nameWithOwner']
            f.write(hashlib.sha256(nwo.encode('utf-8')).hexdigest() + ' 0 0 0 0\n')

def force_close_file(data_lines: List[str], cache_comment: List[str]):
    filename = cache_file_name()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(cache_comment)
        f.writelines(data_lines)

# -------------------------
# Misc counters
# -------------------------

def stars_counter(data_edges):
    total_stars = 0
    for node in data_edges:
        total_stars += node['node']['stargazers']['totalCount']
    return total_stars

def commit_counter(comment_size):
    total_commits = 0
    filename = cache_file_name()
    with open(filename, 'r') as f:
        data = f.readlines()
    data = data[comment_size:]
    for line in data:
        total_commits += int(line.split()[2])
    return total_commits

def svg_overwrite(filename, age_data, commit_data, star_data, repo_data, contrib_data, follower_data, loc_data):
    svg = minidom.parse(filename)
    tspan_indices = {
        'age': 5,
        'repo': 37,
        'contrib': 39,
        'commit': 41,
        'star': 43,
        'follower': 45,
        'loc_total': 47,
        'loc_add': 48,
        'loc_del': 49
    }
    tspan_elements = svg.getElementsByTagName('tspan')
    tspan_elements[tspan_indices['age']].firstChild.data = age_data
    tspan_elements[tspan_indices['repo']].firstChild.data = repo_data
    tspan_elements[tspan_indices['contrib']].firstChild.data = contrib_data
    tspan_elements[tspan_indices['commit']].firstChild.data = commit_data + ' '
    tspan_elements[tspan_indices['star']].firstChild.data = star_data
    tspan_elements[tspan_indices['follower']].firstChild.data = follower_data + ' '
    tspan_elements[tspan_indices['loc_total']].firstChild.data = loc_data[2]
    tspan_elements[tspan_indices['loc_add']].firstChild.data = loc_data[0] + '++'
    tspan_elements[tspan_indices['loc_del']].firstChild.data = loc_data[1] + '--'
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(svg.toxml('utf-8').decode('utf-8'))

# -------------------------
# Main
# -------------------------

if __name__ == '__main__':
    user_idx, org_indices = auto_map_tokens(USER_NAME, ORG_LOGINS)

    user_data, user_time = perf_counter(lambda: user_getter(USER_NAME, user_idx))
    OWNER_ID, acc_date = user_data
    formatter('account data', user_time)

    age_data, age_time = perf_counter(lambda: daily_readme(datetime.datetime(2007, 5, 8)))
    formatter('age calculation', age_time)

    token_for_repo: Dict[str, int] = {}

    user_edges = list_repos_user(user_idx, ['OWNER', 'COLLABORATOR', 'ORGANIZATION_MEMBER'])
    for e in user_edges:
        token_for_repo[e['node']['nameWithOwner']] = user_idx

    org_edges_all: List[Dict[str, Any]] = []
    for j, org in enumerate(ORG_LOGINS):
        idx = org_indices[j]
        edges = list_repos_org(idx, org)
        for e in edges:
            token_for_repo[e['node']['nameWithOwner']] = idx
        org_edges_all.extend(edges)

    all_edges = user_edges + org_edges_all

    total_loc, loc_time = perf_counter(lambda: cache_builder(all_edges, 7, True, token_for_repo))
    QUERY_COUNT['loc_query_total'] += 1
    formatter('LOC (cached)' if total_loc[-1] else 'LOC (no cache)', loc_time)

    commit_data, commit_time = perf_counter(lambda: commit_counter(7))
    commit_data_str = formatter('commit counter', commit_time, commit_data, 1)

    star_user, star_time_user = perf_counter(lambda: graph_repos_stars_user('stars', ['OWNER'], user_idx))
    repo_user, repo_time_user = perf_counter(lambda: graph_repos_stars_user('repos', ['OWNER'], user_idx))

    star_org_total = 0
    repo_org_total = 0
    t_star_org = 0.0
    t_repo_org = 0.0
    for j, org in enumerate(ORG_LOGINS):
        idx = org_indices[j]
        s, ts = perf_counter(lambda o=org, i=idx: graph_repos_stars_org('stars', o, i))
        r, tr = perf_counter(lambda o=org, i=idx: graph_repos_stars_org('repos', o, i))
        star_org_total += s
        repo_org_total += r
        t_star_org += ts
        t_repo_org += tr

    star_data = formatter('star counter', star_time_user + t_star_org, star_user + star_org_total)
    repo_data = formatter('my repositories', repo_time_user + t_repo_org, repo_user + repo_org_total, 1)
    contrib_data = formatter('contributed repos', 0.0, repo_user + repo_org_total, 1)

    follower_data, follower_time = perf_counter(lambda: follower_getter(USER_NAME, user_idx))
    follower_data = formatter('follower counter', follower_time, follower_data, 1)

    for index in range(len(total_loc)-1):
        total_loc[index] = '{:,}'.format(total_loc[index])

    try:
        svg_overwrite('dark_mode.svg', age_data, commit_data_str, star_data, repo_data, contrib_data, follower_data, total_loc[:-1])
        svg_overwrite('light_mode.svg', age_data, commit_data_str, star_data, repo_data, contrib_data, follower_data, total_loc[:-1])
    except Exception:
        pass

    total_time = user_time + age_time + loc_time + (star_time_user + t_star_org) + (repo_time_user + t_repo_org)
