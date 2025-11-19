#ifndef ZMPT101B_MUX_H
#define ZMPT101B_MUX_H

#include <Arduino.h>

class ZMPT101B_MUX {
private:
    int muxChannel;
    float frequency;
    float sensitivity;

    int (*muxReader)(uint8_t);   // function pointer to read MUX ADC

public:
    ZMPT101B_MUX(int channel, float freq, int (*reader)(uint8_t)) {
        muxChannel = channel;
        frequency = freq;
        sensitivity = 500.0;  // default sensitivity
        muxReader = reader;
    }

    void setSensitivity(float sens) {
        sensitivity = sens;
    }

    float getVoltageAC() {
        unsigned long start = micros();
        float sumSq = 0;
        int count = 0;
        unsigned long period_us = (1000000.0 / frequency);  // 20ms for 50Hz

        while (micros() - start < period_us) {
            int adc = muxReader(muxChannel);

            // convert ADC → volts
            float v = adc * (3.3 / 4095.0);

            // remove mid-bias (ZMPT outputs 1.65V center)
            float centered = v - 1.65;

            sumSq += centered * centered;
            count++;
        }

        float rms = sqrt(sumSq / count);

        // scale by sensitivity
        float voltage = rms * sensitivity;

        return voltage;
    }
};

#endif
