#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <string>
#include <sstream>

namespace ci {

namespace detail {
  inline static auto tokenize( std::string str, 
                               std::string delim = " " ) {
    std::istringstream iss(str);
    std::vector<std::string> tokens;

    std::copy( std::istream_iterator<std::string>( iss ),
               std::istream_iterator<std::string>( ),
               std::back_inserter( tokens ) );

    return tokens;
  }
}


  template <size_t N = 64>
  using det_string_t = std::bitset<N>;

  template <size_t N = 64>
  static auto read_wfn_file( std::string fname ) {


    std::ifstream wfn_file( fname );

    size_t ndet_total, norb, nalpha, nbeta;
    {
      std::string line; std::getline( wfn_file, line );
      auto tokens = detail::tokenize( line );
      std::vector<size_t> header_tokens;
      for( auto x : tokens ) 
        header_tokens.push_back(std::stoull(x) );
      ndet_total = header_tokens.at(0);
      norb       = header_tokens.at(1);
      nalpha     = header_tokens.at(2);
      nbeta      = header_tokens.at(3);
    }

    using det_string_type = det_string_t<N>;
    std::vector< det_string_type > alpha_strs, beta_strs;
    alpha_strs.reserve( ndet_total );
    beta_strs.reserve( ndet_total );

    for( std::string line; std::getline(wfn_file, line); ) {
      auto tokens = detail::tokenize( line );
      double coeff    = std::stod( tokens.at(0) );
      std::string det = tokens.at(1);

      std::reverse( det.begin(), det.end() );
      std::string alpha(det), beta(det);

      std::replace( alpha.begin(), alpha.end(), '2', '1' );
      std::replace( alpha.begin(), alpha.end(), 'u', '1' );
      std::replace( alpha.begin(), alpha.end(), 'd', '0' );
      std::replace( beta.begin(),  beta.end(),  '2', '1' );
      std::replace( beta.begin(),  beta.end(),  'd', '1' );
      std::replace( beta.begin(),  beta.end(),  'u', '0' );

      alpha_strs.emplace_back( alpha );
      beta_strs. emplace_back( beta  );
    }

    return std::tuple( alpha_strs, beta_strs );

  }

}
