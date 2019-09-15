#ifndef BASE64_H
#define BASE64_H

#include <string>

using namespace std;

class base64
{
public:
    string encode(unsigned char const*, unsigned int len);
    string decode(string const& s);

private:
    base64();
};

#endif // BASE64_H
