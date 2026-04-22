#include <iostream>
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <unistd.h>

#define LCD_ADDR 0x27
#define LCD_CHR  1
#define LCD_CMD  0

#define LCD_LINE_1 0x80
#define LCD_LINE_2 0xC0
#define LCD_LINE_3 0x94
#define LCD_LINE_4 0xD4

#define En 0b00000100
#define Rw 0b00000010
#define Rs 0b00000001
#define BACKLIGHT 0x08

int fd;

void lcd_toggle_enable(int bits) {
    usleep(500);
    wiringPiI2CWrite(fd, (bits | En));
    usleep(500);
    wiringPiI2CWrite(fd, (bits & ~En));
    usleep(500);
}

void lcd_byte(int bits, int mode) {
    int bits_high = mode | (bits & 0xF0) | BACKLIGHT;
    int bits_low  = mode | ((bits << 4) & 0xF0) | BACKLIGHT;
    wiringPiI2CWrite(fd, bits_high);
    lcd_toggle_enable(bits_high);
    wiringPiI2CWrite(fd, bits_low);
    lcd_toggle_enable(bits_low);
}

void lcd_init() {
    lcd_byte(0x33, LCD_CMD);
    lcd_byte(0x32, LCD_CMD);
    lcd_byte(0x06, LCD_CMD);
    lcd_byte(0x0C, LCD_CMD);
    lcd_byte(0x28, LCD_CMD);
    lcd_byte(0x01, LCD_CMD);
    usleep(500);
}

void lcd_string(const char* message) {
    while (*message) {
        lcd_byte(*message++, LCD_CHR);
    }
}

void lcd_goto(int line) {
    lcd_byte(line, LCD_CMD);
}

void lcd_clear() {
    lcd_byte(0x01, LCD_CMD);
    usleep(500);
}

int main() {
    if (wiringPiSetup() == -1) return 1;

    fd = wiringPiI2CSetup(LCD_ADDR);
    if (fd == -1) return 1;

    lcd_init();
    lcd_clear();

    // LCD2004 có 20 cột (index 0-19)
    // Căn cột:  D=col5, V=col10, X=col15
    //
    // Dòng 1:  "     D    V    X"
    // pos:      01234567890123456789
    //                5    10   15
    //
    // Dòng 2: trống
    //
    // Dòng 3:  "T    2    3    1"
    // Dòng 4:  "LP   0    1    4"

    // Dòng 1 - tiêu đề
    lcd_goto(LCD_LINE_1);
    lcd_string("     D    V    X    ");

    // Dòng 2 - trống
    lcd_goto(LCD_LINE_2);
    lcd_string("                    ");

    // Dòng 3 - hàng T
    lcd_goto(LCD_LINE_3);
    lcd_string("T    2    3    1    ");

    // Dòng 4 - hàng LP
    lcd_goto(LCD_LINE_4);
    lcd_string("LP   0    1    4    ");

    return 0;
}