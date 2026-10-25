"""
Tests of the query builder, they don't need ElasticSearch
"""
import asyncio
import threading

import pytest

from esorm.builder import (Query, Clauses, Clause, Bool, Boosting, ConstantScore, DisMax, FunctionScore,
                           Term, Terms, TermsSet, Range, Exists, Ids, Prefix, Wildcard, Regexp, Fuzzy,
                           Match, MatchPhrase, MatchPhrasePrefix, MatchBoolPrefix, MultiMatch, QueryString,
                           SimpleQueryString, MatchAll, MatchNone, Nested, HasChild, HasParent, ParentId,
                           GeoDistance, GeoBoundingBox, Knn, Raw)
# noinspection PyProtectedMember
from esorm.builder import _current


class TestLeafQueries:
    """
    Query dicts of the leaf queries
    """

    @pytest.mark.parametrize('query, expected', [
        (Term('age', 18), {'term': {'age': 18}}),
        (Term('tag', 'x', boost=2.0, case_insensitive=True, name='n'),
         {'term': {'tag': {'value': 'x', 'case_insensitive': True, 'boost': 2.0, '_name': 'n'}}}),
        (Terms('tag', ['a', 'b']), {'terms': {'tag': ['a', 'b']}}),
        (Terms('tag', ('a', 'b'), boost=1.5), {'terms': {'tag': ['a', 'b'], 'boost': 1.5}}),
        (Terms('tag', {'index': 'i', 'id': '1', 'path': 'tags'}),
         {'terms': {'tag': {'index': 'i', 'id': '1', 'path': 'tags'}}}),
        (TermsSet('skills', ['a', 'b'], minimum_should_match_field='req'),
         {'terms_set': {'skills': {'terms': ['a', 'b'], 'minimum_should_match_field': 'req'}}}),
        (Range('age', gte=18, lt=65), {'range': {'age': {'gte': 18, 'lt': 65}}}),
        (Range('created', gte='now-1d/d', time_zone='+01:00'),
         {'range': {'created': {'gte': 'now-1d/d', 'time_zone': '+01:00'}}}),
        (Exists('email'), {'exists': {'field': 'email'}}),
        (Exists('email', name='has_email'), {'exists': {'field': 'email', '_name': 'has_email'}}),
        (Ids(('1', '2')), {'ids': {'values': ['1', '2']}}),
        (Prefix('name', 'jo'), {'prefix': {'name': 'jo'}}),
        (Prefix('name', 'jo', case_insensitive=True),
         {'prefix': {'name': {'value': 'jo', 'case_insensitive': True}}}),
        (Wildcard('name', 'j*n'), {'wildcard': {'name': 'j*n'}}),
        (Regexp('code', 'ab[0-9]+', flags='ALL'), {'regexp': {'code': {'value': 'ab[0-9]+', 'flags': 'ALL'}}}),
        (Fuzzy('name', 'jonh', fuzziness='AUTO'), {'fuzzy': {'name': {'value': 'jonh', 'fuzziness': 'AUTO'}}}),
        (Match('name', 'John Doe'), {'match': {'name': 'John Doe'}}),
        (Match('name', 'John Doe', operator='and'), {'match': {'name': {'query': 'John Doe', 'operator': 'and'}}}),
        (MatchPhrase('text', 'quick fox', slop=1), {'match_phrase': {'text': {'query': 'quick fox', 'slop': 1}}}),
        (MatchPhrasePrefix('title', 'quick f'), {'match_phrase_prefix': {'title': 'quick f'}}),
        (MatchBoolPrefix('title', 'quick f'), {'match_bool_prefix': {'title': 'quick f'}}),
        (MultiMatch('fox', ('title^3', 'body'), type='best_fields'),
         {'multi_match': {'query': 'fox', 'fields': ['title^3', 'body'], 'type': 'best_fields'}}),
        (MultiMatch('fox'), {'multi_match': {'query': 'fox'}}),
        (QueryString('a OR b', default_field='content'),
         {'query_string': {'query': 'a OR b', 'default_field': 'content'}}),
        (SimpleQueryString('a +b', fields=['title']), {'simple_query_string': {'query': 'a +b', 'fields': ['title']}}),
        (MatchAll(), {'match_all': {}}),
        (MatchAll(boost=1.5), {'match_all': {'boost': 1.5}}),
        (MatchNone(), {'match_none': {}}),
        (ParentId('answer', '1'), {'parent_id': {'type': 'answer', 'id': '1'}}),
        (GeoDistance('loc', {'lat': 47.49, 'lon': 19.04}, '10km'),
         {'geo_distance': {'distance': '10km', 'loc': {'lat': 47.49, 'lon': 19.04}}}),
        (GeoDistance('loc', (19.04, 47.49), 1000, distance_type='plane'),
         {'geo_distance': {'distance': 1000, 'loc': [19.04, 47.49], 'distance_type': 'plane'}}),
        (GeoBoundingBox('loc', top_left={'lat': 47.6, 'lon': 18.9}, bottom_right=(19.2, 47.4)),
         {'geo_bounding_box': {'loc': {'top_left': {'lat': 47.6, 'lon': 18.9}, 'bottom_right': [19.2, 47.4]}}}),
        (Knn('vec', (0.1, 0.2), k=10, num_candidates=100),
         {'knn': {'field': 'vec', 'query_vector': [0.1, 0.2], 'k': 10, 'num_candidates': 100}}),
        (Raw({'span_term': {'text': 'fox'}}), {'span_term': {'text': 'fox'}}),
    ])
    def test_query_dict(self, query: Query, expected: dict):
        assert query.to_query() == expected

    def test_params_can_be_changed(self):
        t = Term('a', 1)
        t.params['boost'] = 2.0
        assert t.to_query() == {'term': {'a': {'value': 1, 'boost': 2.0}}}

    def test_raw_is_not_copied(self):
        d = {'match_all': {}}
        assert Raw(d).to_query() is d

    def test_repr(self):
        assert repr(Term('a', 1)) == "<Term {'term': {'a': 1}}>"
        assert repr(Nested('x')) == "<Nested (the query clause of Nested is not set)>"


class TestCompoundQueries:
    """
    Compound queries with parameters
    """

    def test_bool(self):
        q = Bool(must=[Match('name', 'John'), Range('age', gte=18)],
                 filter=[Term('active', True), {'term': {'country': 'HU'}}],
                 should=Term('tag', 'a'),
                 must_not=[Exists('deleted_at')],
                 minimum_should_match=1, boost=1.5, name='q')
        assert q.to_query() == {'bool': {
            'must': [{'match': {'name': 'John'}}, {'range': {'age': {'gte': 18}}}],
            'filter': [{'term': {'active': True}}, {'term': {'country': 'HU'}}],
            'should': [{'term': {'tag': 'a'}}],
            'must_not': [{'exists': {'field': 'deleted_at'}}],
            'minimum_should_match': 1, 'boost': 1.5, '_name': 'q',
        }}

    def test_empty_bool(self):
        assert Bool().to_query() == {'bool': {}}

    def test_reused_clause(self):
        active = Term('active', True)
        q1 = Bool(filter=[active])
        q2 = Bool(must=[active])
        assert q1.to_query() == {'bool': {'filter': [{'term': {'active': True}}]}}
        assert q2.to_query() == {'bool': {'must': [{'term': {'active': True}}]}}

    def test_clauses_object_is_copied(self):
        q1 = Bool(must=[Term('a', 1)])
        q2 = Bool(must=q1.must)
        q2.must.append(Term('b', 2))
        assert len(q1.must) == 1
        assert len(q2.must) == 2

    def test_invalid_clause(self):
        with pytest.raises(TypeError):
            Bool(must=['x'])  # type: ignore
        with pytest.raises(TypeError):
            Bool().must.append(1)  # type: ignore

    def test_single_query_compounds(self):
        assert Nested('comments', Term('comments.author', 'john'), score_mode='avg').to_query() == {
            'nested': {'path': 'comments', 'query': {'term': {'comments.author': 'john'}}, 'score_mode': 'avg'}}
        assert HasChild('answer', Match('body', 'es'), min_children=2).to_query() == {
            'has_child': {'type': 'answer', 'query': {'match': {'body': 'es'}}, 'min_children': 2}}
        assert HasParent('question', Term('tags', 'py'), score=True).to_query() == {
            'has_parent': {'parent_type': 'question', 'query': {'term': {'tags': 'py'}}, 'score': True}}
        assert ConstantScore(Term('active', True), boost=1.2).to_query() == {
            'constant_score': {'filter': {'term': {'active': True}}, 'boost': 1.2}}
        assert FunctionScore(Match('t', 'es'), weight=2, boost_mode='sum').to_query() == {
            'function_score': {'query': {'match': {'t': 'es'}}, 'weight': 2, 'boost_mode': 'sum'}}
        assert FunctionScore(weight=2).to_query() == {'function_score': {'weight': 2}}
        assert Boosting(Match('t', 'apple'), Match('t', 'pie'), negative_boost=0.5).to_query() == {
            'boosting': {'positive': {'match': {'t': 'apple'}}, 'negative': {'match': {'t': 'pie'}},
                         'negative_boost': 0.5}}
        assert DisMax([Match('a', 'x'), Match('b', 'x')], tie_breaker=0.7).to_query() == {
            'dis_max': {'queries': [{'match': {'a': 'x'}}, {'match': {'b': 'x'}}], 'tie_breaker': 0.7}}

    def test_missing_query(self):
        with pytest.raises(ValueError, match='Nested'):
            Nested('comments').to_query()
        with pytest.raises(ValueError, match='Boosting.negative'):
            Boosting(Term('a', 1), negative_boost=0.5).to_query()

    def test_knn_filter(self):
        assert Knn('vec', [0.1], filter=Term('lang', 'en')).to_query() == {
            'knn': {'field': 'vec', 'query_vector': [0.1], 'filter': {'term': {'lang': 'en'}}}}
        assert Knn('vec', [0.1], filter=[Term('a', 1), Term('b', 2)]).to_query() == {
            'knn': {'field': 'vec', 'query_vector': [0.1], 'filter': [{'term': {'a': 1}}, {'term': {'b': 2}}]}}


class TestWithBlocks:
    """
    Building queries with with blocks
    """

    def test_same_as_parameters(self):
        q1 = Bool(must=[Match('name', 'John'), Range('age', gte=18)], filter=[Term('active', True)],
                  must_not=[Exists('deleted_at')])
        with Bool() as q2:
            with q2.must:
                Match('name', 'John')
                Range('age', gte=18)
            with q2.filter:
                Term('active', True)
            with q2.must_not:
                Exists('deleted_at')
        assert q1.to_query() == q2.to_query()

    def test_bool_block_is_must(self):
        with Bool() as q:
            Term('a', 1)
            with q.filter:
                Term('b', 2)
            Term('c', 3)
        assert q.to_query() == {'bool': {'must': [{'term': {'a': 1}}, {'term': {'c': 3}}],
                                         'filter': [{'term': {'b': 2}}]}}

    def test_deep_nesting(self):
        with Bool() as q:
            Term('active', True)
            with q.should:
                Term('tag', 'python')
                Bool(must=[Term('tag', 'es'), Range('stars', gte=100)])
                with Nested('comments'):
                    Match('comments.text', 'great')
                with Bool() as inner:
                    with inner.must_not:
                        Exists('x')
            q.minimum_should_match = 1
        assert q.to_query() == {'bool': {
            'must': [{'term': {'active': True}}],
            'should': [
                {'term': {'tag': 'python'}},
                {'bool': {'must': [{'term': {'tag': 'es'}}, {'range': {'stars': {'gte': 100}}}]}},
                {'nested': {'path': 'comments', 'query': {'match': {'comments.text': 'great'}}}},
                {'bool': {'must_not': [{'exists': {'field': 'x'}}]}},
            ],
            'minimum_should_match': 1,
        }}

    def test_adoption_keeps_order(self):
        with Bool() as q:
            Term('a', 1)
            Bool(should=[Term('b', 2), Term('c', 3)])
            Term('d', 4)
        assert q.to_query() == {'bool': {'must': [
            {'term': {'a': 1}},
            {'bool': {'should': [{'term': {'b': 2}}, {'term': {'c': 3}}]}},
            {'term': {'d': 4}},
        ]}}

    def test_append_moves(self):
        with Bool() as q:
            with q.should:
                t = Term('a', 1)
                q.filter.append(t)  # moved from should to filter
                Term('b', 2)
                q.should.append(Term('c', 3))  # not duplicated
        assert q.to_query() == {'bool': {'filter': [{'term': {'a': 1}}],
                                         'should': [{'term': {'b': 2}}, {'term': {'c': 3}}]}}

    def test_not_moved_after_the_block(self):
        with Bool() as q:
            t = Term('a', 1)
        other = Bool(filter=[t])
        assert q.to_query() == {'bool': {'must': [{'term': {'a': 1}}]}}
        assert other.to_query() == {'bool': {'filter': [{'term': {'a': 1}}]}}

    def test_single_query_blocks(self):
        with Nested('comments') as n:
            Term('comments.author', 'john')
            Range('comments.stars', gte=4)
        assert n.to_query() == {'nested': {'path': 'comments', 'query': {'bool': {'must': [
            {'term': {'comments.author': 'john'}}, {'range': {'comments.stars': {'gte': 4}}}]}}}}

        with Boosting(negative_boost=0.5) as b:
            Match('t', 'apple')  # positive
            with b.negative:
                Match('t', 'pie')
        assert b.to_query() == {'boosting': {'positive': {'match': {'t': 'apple'}},
                                             'negative': {'match': {'t': 'pie'}}, 'negative_boost': 0.5}}

        with DisMax() as d:
            Match('a', 'x')
            Match('b', 'x')
        assert d.to_query() == {'dis_max': {'queries': [{'match': {'a': 'x'}}, {'match': {'b': 'x'}}]}}

        with ConstantScore() as c:
            Term('a', 1)
        assert c.to_query() == {'constant_score': {'filter': {'term': {'a': 1}}}}

        with Knn('vec', [0.1], k=5) as k:
            Term('lang', 'en')
        assert k.to_query() == {'knn': {'field': 'vec', 'query_vector': [0.1], 'k': 5,
                                        'filter': {'term': {'lang': 'en'}}}}

    def test_clause_set(self):
        c = Clause()
        assert not c
        c.set(Term('a', 1))
        assert c.render('test') == {'term': {'a': 1}}

    def test_raw_in_block(self):
        with Bool() as q:
            Raw({'span_term': {'text': 'fox'}})
        assert q.to_query() == {'bool': {'must': [{'span_term': {'text': 'fox'}}]}}

    def test_outside_of_blocks(self):
        Term('a', 1)
        assert _current.get() is None

    def test_exception(self):
        with pytest.raises(RuntimeError):
            with Bool() as q:
                with q.filter:
                    Term('a', 1)
                    raise RuntimeError()
        assert _current.get() is None
        Term('b', 2)  # not collected
        assert q.to_query() == {'bool': {'filter': [{'term': {'a': 1}}]}}

    async def test_async_tasks(self):
        async def build(n: int) -> dict:
            with Bool() as b:
                for i in range(3):
                    Term('t', n * 10 + i)
                    await asyncio.sleep(0)
            return b.to_query()

        r1, r2 = await asyncio.gather(build(1), build(2))
        assert r1 == {'bool': {'must': [{'term': {'t': 10}}, {'term': {'t': 11}}, {'term': {'t': 12}}]}}
        assert r2 == {'bool': {'must': [{'term': {'t': 20}}, {'term': {'t': 21}}, {'term': {'t': 22}}]}}

    def test_threads(self):
        results = {}
        barrier = threading.Barrier(2)

        def build(n: int):
            with Bool() as b:
                barrier.wait()
                Term('t', n)
                barrier.wait()
            results[n] = b.to_query()

        threads = [threading.Thread(target=build, args=(n,)) for n in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == {n: {'bool': {'must': [{'term': {'t': n}}]}} for n in (1, 2)}

    async def test_child_task(self):
        """ A task created in a block inherits the block, but its queries are not collected by it """
        ready = asyncio.Event()

        async def worker():
            Term('loose', 1)
            await ready.wait()
            with Bool() as child:
                Term('child', 1)
            Term('loose', 2)
            return child

        with Bool() as parent:
            Term('parent', 1)
            task = asyncio.create_task(worker())
            await asyncio.sleep(0)  # the worker starts while the block is active
        ready.set()
        child = await task
        assert parent.to_query() == {'bool': {'must': [{'term': {'parent': 1}}]}}
        assert child.to_query() == {'bool': {'must': [{'term': {'child': 1}}]}}

    def test_child_thread(self):
        """ Threads started in a block (even with the context copied) don't collect into it """
        import contextvars
        results = {}

        def worker():
            Term('loose', 1)
            with Bool() as child:
                Term('child', 1)
            results['child'] = child.to_query()

        with Bool() as parent:
            Term('parent', 1)
            ctx = contextvars.copy_context()
            t = threading.Thread(target=ctx.run, args=(worker,))
            t.start()
            t.join()
        assert parent.to_query() == {'bool': {'must': [{'term': {'parent': 1}}]}}
        assert results['child'] == {'bool': {'must': [{'term': {'child': 1}}]}}


class TestOperators:
    """
    &, |, ~
    """

    def test_and(self):
        assert (Term('a', 1) & Term('b', 2) & Term('c', 3)).to_query() == {'bool': {'must': [
            {'term': {'a': 1}}, {'term': {'b': 2}}, {'term': {'c': 3}}]}}

    def test_or(self):
        assert (Term('a', 1) | Term('b', 2) | Term('c', 3)).to_query() == {'bool': {'should': [
            {'term': {'a': 1}}, {'term': {'b': 2}}, {'term': {'c': 3}}], 'minimum_should_match': 1}}

    def test_not(self):
        assert (~Exists('x')).to_query() == {'bool': {'must_not': [{'exists': {'field': 'x'}}]}}

    def test_mixed(self):
        q = (Term('a', 1) | Term('b', 2)) & ~Term('c', 3)
        assert q.to_query() == {'bool': {'must': [
            {'bool': {'should': [{'term': {'a': 1}}, {'term': {'b': 2}}], 'minimum_should_match': 1}},
            {'bool': {'must_not': [{'term': {'c': 3}}]}},
        ]}}

    def test_no_flattening_with_params(self):
        a = Bool(must=[Term('a', 1)], boost=2.0)
        assert (a & Term('b', 2)).to_query() == {'bool': {'must': [
            {'bool': {'must': [{'term': {'a': 1}}], 'boost': 2.0}}, {'term': {'b': 2}}]}}
        f = Bool(must=[Term('a', 1)], filter=[Term('x', 1)])
        assert len((f & Term('b', 2)).must) == 2

    def test_or_with_empty_bool(self):
        """ An empty bool matches every document, so it can't be dropped from an or """
        for q in (Bool() | Term('a', 1), Term('a', 1) | Bool()):
            assert len(q.should) == 2
        assert (Bool() | Term('a', 1)).to_query() == {'bool': {'should': [
            {'bool': {}}, {'term': {'a': 1}}], 'minimum_should_match': 1}}
        assert (Term('a', 1) | Bool()).to_query() == {'bool': {'should': [
            {'term': {'a': 1}}, {'bool': {}}], 'minimum_should_match': 1}}
        assert (Bool(minimum_should_match=1) | Term('a', 1)).to_query() == {'bool': {'should': [
            {'bool': {'minimum_should_match': 1}}, {'term': {'a': 1}}], 'minimum_should_match': 1}}
        # An empty bool in an and is a match all, dropping it doesn't change the meaning
        assert (Bool() & Term('a', 1)).to_query() == {'bool': {'must': [{'term': {'a': 1}}]}}

    def test_operands_are_not_changed(self):
        a = Term('a', 1) & Term('b', 2)
        b = a & Term('c', 3)
        assert len(a.must) == 2
        assert len(b.must) == 3

    def test_dicts(self):
        assert (Term('a', 1) & {'match_all': {}}).to_query() == {'bool': {'must': [
            {'term': {'a': 1}}, {'match_all': {}}]}}
        assert ({'match_all': {}} | Term('a', 1)).to_query() == {'bool': {'should': [
            {'match_all': {}}, {'term': {'a': 1}}], 'minimum_should_match': 1}}

    def test_in_with_block(self):
        with Bool() as q:
            with q.should:
                Term('a', 1) & Term('b', 2)
                Term('c', 3)
        assert q.to_query() == {'bool': {'should': [
            {'bool': {'must': [{'term': {'a': 1}}, {'term': {'b': 2}}]}}, {'term': {'c': 3}}]}}


class TestClauses:
    """
    Clauses as a list
    """

    def test_list_methods(self):
        q = Bool()
        assert not q.must
        q.must.append(Term('a', 1))
        q.must += Term('b', 2)
        q.must += [Term('c', 3), {'term': {'d': 4}}]
        q.must.extend([Term('e', 5)])
        assert len(q.must) == 5
        assert q.must
        assert isinstance(q.must[0], Term)
        assert len(q.must[1:3]) == 2
        assert [type(c).__name__ for c in q.must] == ['Term', 'Term', 'Term', 'dict', 'Term']
        q.must.clear()
        assert q.to_query() == {'bool': {}}

    def test_extend_from_active_clauses(self):
        """ Moving the clauses of an active block doesn't skip any of them """
        with Bool() as parent:
            Term('a', 1)
            Term('b', 2)
            Term('c', 3)
            child = Bool(filter=parent.must)
        assert child.to_query() == {'bool': {'filter': [
            {'term': {'a': 1}}, {'term': {'b': 2}}, {'term': {'c': 3}}]}}
        assert parent.to_query() == {'bool': {'must': [child.to_query()]}}

        other = Bool()
        with Bool() as parent:
            Term('a', 1)
            Term('b', 2)
            other.must.extend(parent.must)
        assert len(other.must) == 2
        assert not parent.must

    def test_self_extend(self):
        q = Bool(must=[Term('a', 1), Term('b', 2)])
        q.must += q.must
        q.must.extend(q.must)
        assert len(q.must) == 8

    def test_assign(self):
        q = Bool(must=[Term('a', 1)])
        q.should = Clauses([Term('b', 2)])
        assert q.to_query() == {'bool': {'must': [{'term': {'a': 1}}], 'should': [{'term': {'b': 2}}]}}

    def test_with_as(self):
        q = Bool()
        with q.filter as f:
            Term('a', 1)
            assert len(f) == 1
        assert q.to_query() == {'bool': {'filter': [{'term': {'a': 1}}]}}

    def test_dynamic(self):
        def build(name, min_age, tags):
            q = Bool()
            if name:
                q.must.append(Match('name', name))
            if min_age is not None:
                q.filter += Range('age', gte=min_age)
            if tags:
                q.filter += Terms('tag', tags)
            return q.to_query()

        assert build(None, None, []) == {'bool': {}}
        assert build('John', 18, ['a']) == {'bool': {'must': [{'match': {'name': 'John'}}], 'filter': [
            {'range': {'age': {'gte': 18}}}, {'terms': {'tag': ['a']}}]}}


class TestCustomQuery:
    """
    A custom query clause
    """

    def test_custom(self):
        class Published(Query):
            def __init__(self, value: bool = True):
                self.value = value
                super().__init__()

            def to_query(self):
                return {'term': {'published': self.value}}

        with Bool() as q:
            Published()
            Term('a', 1) | Published(False)
        assert q.to_query() == {'bool': {'must': [
            {'term': {'published': True}},
            {'bool': {'should': [{'term': {'a': 1}}, {'term': {'published': False}}], 'minimum_should_match': 1}},
        ]}}
