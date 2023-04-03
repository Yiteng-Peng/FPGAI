CC=gcc
CFLAGS=-I.
DEPS=dataset.h LeNet.h Quant.h Load.h

ODIR=obj
_OBJ = inference.o LeNet.o Quant.o Load.o dataset.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

lenet: $(OBJ)
	$(CC) -o $@ $^
