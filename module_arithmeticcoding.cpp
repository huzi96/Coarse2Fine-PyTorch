// This file is being made available under the BSD License.  
// Copyright (c) 2021 Yueyu Hu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <cstring>
using namespace std;

// typedef unsigned long comint;
typedef __int128 comint;
const comint STATE_SIZE = 64;
const comint MAX_RANGE = (comint)1 << STATE_SIZE;
const comint MIN_RANGE = (MAX_RANGE >> 2) + 2;
const comint MAX_TOTAL = MIN_RANGE;
const comint MASK = MAX_RANGE - 1;
const comint TOP_MASK = MAX_RANGE >> 1;
const comint SECOND_MASK = TOP_MASK >> 1;

typedef unsigned int uint32;
typedef unsigned char uint8;

class BitInputStream
{
public:
    FILE* input;
    unsigned char currentbyte;
    int numbitsremaining;
    BitInputStream(FILE* inp) {
        input = inp;
        currentbyte = 0;
        numbitsremaining = 0;
    }

    char read() {
        if(currentbyte == -1) {
            return -1;
        }
        if(numbitsremaining == 0) {
            unsigned char temp;
            int nread = fread(&temp, 1, 1, input);
            if (nread == 0) {
                currentbyte = -1;
                return -1;
            }
            currentbyte = temp;
            numbitsremaining = 8;
        }
        numbitsremaining--;
        return (currentbyte >> numbitsremaining) & 1;
    }

    void close() {
        fclose(input);
        currentbyte = -1;
        numbitsremaining = 0;
    }
    
};

class BitOutputStream
{
public:
    FILE *output;
    int currentbyte;
    int numbitsfilled;
    BitOutputStream(FILE *out) {
        output = out;
        numbitsfilled = 0;
        currentbyte = 0;
    }

    void write(uint8 b) {
        if (b > 1) {
            fprintf(stderr, "Argument must be 0 or 1");
        }
        currentbyte = (currentbyte << 1) | b;
        numbitsfilled += 1;
        if (numbitsfilled == 8) {
            uint8 towrite = currentbyte;
            fwrite(&towrite, 1, 1, output);
            numbitsfilled = 0;
            currentbyte = 0;
        }
    }

    void close() {
        while(numbitsfilled != 0) write(0);
        fclose(output);
    }
};

uint32 FL_MASK1 = ~(0x3FFF);
uint32 FL_MASK2 = ~(0x7FFF);

void set_FL_MASK(int level, int no)
{
    uint32 FL_MASK = (1<<level);
    FL_MASK -= 1;
    if (no == 1)
        FL_MASK1 = ~(FL_MASK);
    else
        FL_MASK2 = ~(FL_MASK);
}

float convert(float* p, uint32 FL_MASK)
{
    uint32 bits = *((uint32 *)p);
    bits = bits & FL_MASK;
    float ret = 0;
    *((uint32 *)(&ret)) = bits;
    return ret;
}

class ModelFrequencyTable
{
public:
    int mul_factor;
    int num_symbols;
    int _EOF;
    float mu_val;
    float sigma_val;
    float TINY;
    int total;
    float *cumulative;
    ModelFrequencyTable(float& mu_val_, float& sigma_val_):mul_factor(10000000),num_symbols(1025)
    {
        mu_val = convert(&mu_val_, FL_MASK1);
        sigma_val = convert(&sigma_val_, FL_MASK2);
        _EOF = num_symbols - 1;
        TINY = 1e-10;
        total = mul_factor + 1025;
        cumulative = NULL;
    }

    void set_mu(float mu_val_) {mu_val = mu_val_;}

    void set_sigma(float sigma_val_) {sigma_val = sigma_val_;}

    int get_symbol_limit() {return num_symbols;}

    int get(comint symbol) {
        _check_symbol(symbol);
        if (symbol == _EOF) return 1;
        else {
            float c2 = 0.5 * (1 + erf((symbol + 0.5 - mu_val) /  ((sigma_val + TINY) * sqrt(2) )));
            float c1 = 0.5 * (1 + erf((symbol - 0.5 - mu_val) /  ((sigma_val + TINY) * sqrt(2) )));
            int freq = (int)(floor((c2 - c1) * mul_factor) + 1);
            return freq;
        }
    }

    int get_total() {return total;}

    comint get_low(comint symbol) {
        _check_symbol(symbol);
        float c = 0.5 * (1 + erf(((symbol-1) + 0.5 - mu_val) / ((sigma_val + TINY) * sqrt(2))));
        comint _c = (int)(floor(c * mul_factor) + symbol);
        return _c;
    }

    comint get_high(comint symbol) {
        _check_symbol(symbol);
        float c = 0.5 * (1 + erf(((symbol) + 0.5 - mu_val) / ((sigma_val + TINY) * sqrt(2))));
        comint _c = (int)(floor(c * mul_factor) + symbol+1);
        return _c;
    }

    void _check_symbol(comint symbol) {
        if (0 <= symbol && symbol < get_symbol_limit())
        return;
        else {
            fprintf(stderr, "Symbol out of range\n");
            exit(-1);
        }
    }
};

class ArithmeticCoderBase
{
public:

    comint _low, _high;
    ArithmeticCoderBase():_low(0),_high(MASK) {}

    virtual void shift() = 0;
    virtual void underflow() = 0;

    virtual void update(ModelFrequencyTable *freqs, comint symbol) {
        comint low=_low, high=_high;
        if ( (low >= high) || ((low & MASK) != low) || ((high & MASK) != high) ) {
            fprintf(stderr, "Low or high out of range\n");
            exit(-1);
        }
        comint range_ = high - low + 1;
        if (!(MIN_RANGE <= range_ && range_ <= MAX_RANGE)) {
            fprintf(stderr, "Range out of range\n");
            exit(-1);
        }

        comint total = freqs -> get_total();
        comint symlow = freqs -> get_low(symbol);
        comint symhigh = freqs -> get_high(symbol);
        if (symlow == symhigh) {
            for (int idx=symbol-20; idx<symbol+20; idx++) {
                fprintf(stderr, "----------------\n");
                fprintf(stderr, "symbol: %d", idx);
                fprintf(stderr, "symlow: %ld", freqs->get_low(idx));
                fprintf(stderr, "symhigh: %ld", freqs->get_high(idx));
                fprintf(stderr, "----------------\n");
                fprintf(stderr, "Symbol has zero frequency\n");
                exit(-1);
            }
        }

        if (total > MAX_TOTAL) {
            fprintf(stderr, "Cannot code symbol because total is too large\n");
            exit(-1);
        }
        comint newlow = low + symlow * range_ / total;
        comint newhigh = low + symhigh * range_ / total - 1;
        _low = newlow;
        _high = newhigh;

        int ccc;

        while ( ((_low ^ _high) & TOP_MASK) == 0 ) {
            this -> shift();
            _low = (_low << 1) & MASK;
            _high = ((_high << 1) & MASK) | 1;
        }

        while ((( _low & ~_high & SECOND_MASK )) != 0) {
            this -> underflow();
            _low = (_low << 1) & (MASK >> 1);
            _high = ((_high << 1) & (MASK >> 1)) | TOP_MASK | 1;
        }
        return;
    }
};

class ArithmeticEncoder:ArithmeticCoderBase
{
public:
    BitOutputStream* output;
    comint num_underflow;
    ArithmeticEncoder(BitOutputStream* bitout) {
        output = bitout;
        num_underflow = 0;
    }

    void write(ModelFrequencyTable* freqs, comint symbol) {
        update(freqs, symbol);
    }

    void finish() {
        output->write(1);
    }

    void shift() {
        comint bit = _low >> (STATE_SIZE - 1);
        output->write(bit);
        for (int i = 0; i < num_underflow; i++) {
            output -> write(bit ^ 1);
        }
        num_underflow = 0;
    }

    void underflow() {
        num_underflow++;
    }
};

class ArithmeticDecoder:ArithmeticCoderBase
{
public:
    BitInputStream *input;
    comint code;
    ArithmeticDecoder(BitInputStream *bitin) {
        input = bitin;
        code = 0;
        for (int i = 0; i < STATE_SIZE; i++) {
            code = (code << 1) | read_code_bit();
        }
    }

    comint read(ModelFrequencyTable *freqs) {
        comint total = freqs -> get_total();
        if (total > MAX_TOTAL) {
            fprintf(stderr, "Cannot decode symbol becaise total is too large");
            exit(-1);
        }

        comint range = _high - _low + 1;
        comint offset = code - _low;
        comint value = (comint) (floor(((double)offset / range) * total + ((double)total / range) - ( 1. / range)));

        if (! (floor(((double) value / total) * ((double) range / total)) <= offset) ) {
            fprintf(stderr, "offset: %ld", offset);
            fprintf(stderr, "range: %ld", range);
            fprintf(stderr, "value: %ld", value);
            fprintf(stderr, "total: %ld", total);
        }
        assert(floor(((double)value/total) * ((double)range/total)) <= offset);

        if (!(0 <= value && value < total)) {
            fprintf(stderr, "offset: %ld", offset);
            fprintf(stderr, "range: %ld", range);
            fprintf(stderr, "value: %ld", value);
            fprintf(stderr, "total: %ld", total);
        }
        assert(0 <= value && value < total);

        comint start = 0;
        comint end = freqs -> get_symbol_limit();
        while (end - start > 1) {
            comint middle = (start + end) >> 1;
            if (freqs->get_low(middle) > value) {
                end = middle;
            }
            else {
                start = middle;
            }
        }
        assert(start + 1 == end);

        comint symbol = start;

        if (!(freqs -> get_low(symbol) * range / total <= offset && offset < freqs -> get_high(symbol) * range / total)) {
            fprintf(stderr, "symbol: %ld\n", symbol);
            fprintf(stderr, "freqs.get_symbol_limit(): %d\n", freqs -> get_symbol_limit());
            fprintf(stderr, "freqs.get_low(symbol): %ld\n", freqs -> get_low(symbol));
            fprintf(stderr, "freqs.get_high(symbol): %ld\n", freqs -> get_high(symbol));
            fprintf(stderr, "offset: %ld\n", offset);
            fprintf(stderr, "range: %ld\n", range);
            fprintf(stderr, "value: %ld\n", value);
            fprintf(stderr, "total: %ld\n", total);
        }
        assert(freqs->get_low(symbol) * range / total <= offset && offset < freqs->get_high(symbol) * range / total);

        update(freqs, symbol);
        if (!(_low <= code && code <= _high)) {
            fprintf(stderr, "Code out of range\n");
            exit(-1);
        }
        return symbol;
    }

    void shift() {
        unsigned char bit = read_code_bit();
        code = ((code << 1) & MASK ) | bit;
    }

    void underflow() {
        code = ((code & TOP_MASK) | (code << 1) & (MASK >> 1)) | read_code_bit();
    }

    comint read_code_bit() {
        comint temp = input->read();
        if (temp == -1) temp = 0;
        return temp;
    }

    comint getlow() {return _low;}
    comint gethigh() {return _high;}
};

int main(int argc, char *argv[])
{
    char *mode = argv[1];
    int level1 = atoi(argv[2]);
    int level2 = atoi(argv[3]);
    // fprintf(stderr, "%d %d\n", level1, level2);
    set_FL_MASK(level1, 1);
    set_FL_MASK(level2, 2);
    if (mode[0] == 'e') {
        long len;
        fread(&len, 8, 1, stdin);
        short *coeff = new short[len];
        float *mu = new float[len];
        float *sigma = new float[len];
        fread(coeff, 2, len, stdin);
        fread(mu, 4, len, stdin);
        fread(sigma, 4, len, stdin);
        
        BitOutputStream bs(stdout);
        ArithmeticEncoder enc(&bs);

        for (int i=0; i<len; i++) {
            float mu_val = mu[i];
            float sigma_val = sigma[i];
            ModelFrequencyTable freq(mu_val, sigma_val);
            enc.write(&freq, coeff[i]);
        }
        float mu_val = 255;
        float sigma_val = 1;
        ModelFrequencyTable freq(mu_val, sigma_val);
        enc.write(&freq, 512);
        enc.finish();
        delete [] coeff;
        delete [] mu;
        delete [] sigma;
    }
    else if (mode[0] == 'd') {
        long blen, len;
        // fread(&blen, 8, 1, stdin);
        fread(&len, 8, 1, stdin);
        // unsigned char *buf = new unsigned char[blen];
        float *mu = new float[len];
        float *sigma = new float[len];
        // fread(buf, 1, blen, stdin);
        fread(mu, 4, len, stdin);
        fread(sigma, 4, len, stdin);
        
        BitInputStream bs(stdin);
        ArithmeticDecoder dec(&bs);

        short *coeff = new short[len];
        for (int i=0; i<len; i++) {
            float mu_val = mu[i];
            float sigma_val = sigma[i];
            ModelFrequencyTable freq(mu_val, sigma_val);
            short symbol = dec.read(&freq);
            coeff[i] = symbol;
        }
        
        fwrite(coeff, 2, len, stdout);
        delete [] coeff;
        delete [] mu;
        delete [] sigma;
    }
}
