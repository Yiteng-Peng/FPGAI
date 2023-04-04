CC=gcc
CFLAGS=-I.

DEPS=dataset.h LeNet.h Quant.h Load.h ResNet.h
ODIR=obj
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

_OBJ = inference.o LeNet.o Quant.o Load.o dataset.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
lenet: $(OBJ)
	$(CC) -o $@ $^

_OBJ = inference.o ResNet.o Quant.o Load.o dataset.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
resnet: $(OBJ)
	$(CC) -o $@ $^
