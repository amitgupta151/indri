


//
// Random Walk Model
//
// 24 Mar 2015 -- Amit Gupta amitgupta151@gmail.com
//


#include <string>
#include <vector>
#include "indri/TermFieldStatistics.hpp"
#include "indri/TermScoreFunction.hpp"
#include "indri/TermScoreFunctionFactory.hpp"
#include "indri/HashTable.hpp"
#include "indri/greedy_vector"
#include "indri/QueryEnvironment.hpp"

namespace indri {
  namespace query {
    class RandomWalkModel {
    public:
      struct Gram {
        std::vector<std::string> terms;
        double weight;

        std::string term_string() {
        	std::string ans = "";
        	for (int i = 0 ; i < terms.size(); i++) ans += " " + terms[i];
        	if (ans.length() > 0) return ans.substr(1);
        	else return ans;
        }


        struct hash {
          int operator() ( const Gram* one ) const {
            indri::utility::GenericHash<const char*> h;
            int accumulator = 0;

            for( size_t i=0; i<one->terms.size(); i++ ) {
              accumulator *= 7;
              accumulator += h( one->terms[i].c_str() );
            }

            return accumulator;
          }
        };

        struct weight_greater {
          bool operator() ( const Gram* o, const Gram* t ) const {
            return t->weight < o->weight;
          }
        };

        struct string_comparator {
          int operator() ( const Gram* o, const Gram* t ) const {
            const Gram& one = *o;
            const Gram& two = *t;

            if( one.terms.size() != two.terms.size() ) {
              if( one.terms.size() < two.terms.size() ) {
                return 1;
              } else {
                return -1;
              }
            }

            for( size_t i=0; i<one.terms.size(); i++ ) {
              const std::string& oneString = one.terms[i];
              const std::string& twoString = two.terms[i];

              if( oneString != twoString ) {
                if( oneString < twoString )
                  return -1;
                else
                  return 1;
              }
            }

            return 0;
          }
        };
      };


     private:
      struct GramCounts {
        Gram gram;
        indri::utility::greedy_vector< std::pair< int, int > > counts;
      };


      indri::api::QueryEnvironment& _environment;
      int _maxGrams;
      std::string _smoothing;
      int _documents;

      typedef indri::utility::greedy_vector< std::pair< Gram*, size_t > > VectorGramCounts;
      typedef indri::utility::HashTable< size_t, VectorGramCounts> CountVectorGram;
      typedef indri::utility::HashTable< Gram*, GramCounts*, Gram::hash, Gram::string_comparator > HGram;
      typedef indri::utility::HashTable< Gram*, size_t, Gram::hash, Gram::string_comparator > HGramCount;
      typedef indri::utility::HashTable< size_t,Gram*  > HCountGram;

      std::vector<std::string> _queryGrams;
      // gram table :- hash (gram -> list ( doc_id, count))
      HGram _gramTable;
      // gram counts :- hash (gram -> count)
      //gram id :- hash (gram -> id)
      HGramCount _gramCounts,_gramIds;
      // _docIdGrams :- hash (docid -> list (gram , count) )
      CountVectorGram _docIdGrams;
      // id gram :- hash (id -> gram)
      HCountGram _idGrams;

      std::vector<indri::api::ScoredExtentResult> _results;
      std::vector<lemur::api::DOCID_T> _documentIDs;
      std::vector<Gram*> _grams;
      std::vector<indri::api::DocumentVector*> _vectors;
      vector<std::string> indri::query::RandomWalkModel::find_query_grams(std::string ans);
      vector<std::string> indri::query::RandomWalkModel::tokenize_string(std::string ans);

      void _countGrams();
      void _scoreGrams();
      void _sortGrams();
      void _extractDocuments();
      void _buildCoocMatrix();

      // id , id -> cooccur
      size_t ** _cooccurMatrix;


    public:
      RandomWalkModel( indri::api::QueryEnvironment& environment,
                      const std::string& smoothing,\
                      int maxGrams,
                      int documents );
      ~RandomWalkModel();

      void generate( const std::string& query );
      // generate from an existing result set
      void generate( const std::string &query , const std::vector<indri::api::ScoredExtentResult>& results );
      const std::vector<indri::api::ScoredExtentResult>& getQueryResults() const;
      const std::vector<Gram*>& getGrams() const;
    };
  }
}
