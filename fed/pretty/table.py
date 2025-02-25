import math
import collections.abc as collections


class Percent(float):
    def __format__(self, _):
        return '{:.2f}%'.format(self * 100)


def unit(value, asint=False, base=1024):
    if value == 0:
        return value
    exp = math.floor(math.log(abs(value), base))
    value = value / 1024 ** exp
    value = int(value) if asint else f'{value:.2f}'
    return f'{value}{" KMGTP"[exp]}'.replace(' ', '')


class Bits(int):
    def __format__(self, mode):
        value = self
        is_bytes = 'b' not in mode
        if is_bytes:
            value = math.ceil(self / 8.0)
        suffix = 'B' if is_bytes else 'b'
        if 'i' in mode:
            suffix = ''
        return f'{unit(value)}{suffix}'

    def __str__(self):
        return '{}'.format(self)

    def __repr__(self):
        return str(self)


class _Unknown(int):
    def __format__(self, mode):
        return ''

    def __str__(self):
        return ''
    __repr__ = __str__


unknown = _Unknown(0)


class Table(collections.Sequence):
    styles = {
        'mayo': {
            'top': ['+-', '-+-', '-', '-+'],
            'bottom': ['+-', '-+-', '-', '-+'],
            'middle': ['+-', '-+-', '-', '-+'],
            'column': ['| ', ' | ', ' |']
        },
        'github': {
            'top': None,
            'bottom': None,
            'middle': ['', ' | ', '-', ''],
            'column': ['', ' | ', ''],
        },
    }

    def __init__(self, headers, formatters=None, style='github'):
        super().__init__()
        self.headers = list(headers)
        self.style = style
        self._empty_formatters = {h: None for h in headers}
        self._formatters = formatters or self._empty_formatters
        self._rows = []
        self._rules = []
        self._footers = {}
        self.add_rule()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return self._rows[row][self.headers.index(col)]
        return self._rows[index]

    def __len__(self):
        return len(self._rows)

    @property
    def num_columns(self):
        return len(self.headers)

    @classmethod
    def from_namedtuples(cls, tuples, formatters=None):
        table = cls(tuples[0]._fields, formatters)
        table.add_rows(tuples)
        return table

    @classmethod
    def from_dictionaries(cls, dictionaries, formatters=None):
        headers = list(dictionaries[0])
        table = cls(headers, formatters)
        for each in dictionaries:
            table.add_row([each[h] for h in headers])
        return table

    def add_row(self, row):
        if isinstance(row, collections.Mapping):
            row = [row[h] for h in self.headers]
        if len(row) != len(self.headers):
            raise ValueError(
                'Number of columns of row {!r} does not match headers {!r}.'
                .format(row, self.headers))
        self._rows.append(list(row))

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_rule(self):
        self._rules.append(len(self._rows))

    def add_column(self, name, func, formatter=None):
        for row_idx, row in enumerate(self._rows):
            new = func(row_idx)
            row.append(new)
        self.headers.append(name)
        self._formatters[name] = formatter

    def footer_sum(self, column):
        self._footers[column] = {'method': 'sum'}

    def footer_max(self, column):
        self._footers[column] = {'method': 'max'}

    def footer_mean(self, column, weights=None):
        self._footers[column] = {'method': 'mean', 'weights': weights}

    def footer_value(self, column, value):
        self._footers[column] = {'method': 'value', 'value': value}

    def get_column(self, name):
        try:
            col_idx = self.headers.index(name)
        except ValueError as e:
            desc = 'Unable to find header named {!r}.'.format(name)
            raise KeyError(desc) from e
        return [row[col_idx] for row in self._rows]

    def _format_value(self, value, formatter=None, width=None):
        if value is None:
            value = ''
        elif formatter:
            if isinstance(formatter, collections.Callable):
                value = formatter(value, width)
            else:
                value = formatter.format(value, width=width)
        elif isinstance(value, int):
            value = '{:{width},}'.format(value, width=width or 0)
        elif isinstance(value, float):
            value = '{:{width}.{prec}}'.format(
                value, width=width or 0, prec=3)
        elif isinstance(value, (list, tuple)):
            value = ', '.join(self._format_value(v) for v in value)
        else:
            value = str(value)
        if width:
            value = '{:{width}}'.format(value, width=width)
            if len(value) > width:
                value = value[:width - 1] + '…'
        return value

    def _format_row(self, row, formatters=None, widths=None):
        formatters = formatters or self._formatters
        widths = widths or [None] * len(self.headers)
        new_row = []
        for h, x, width in zip(self.headers, row, widths):
            x = self._format_value(x, formatters[h], width)
            new_row.append(x)
        return new_row

    def _get_footers(self):
        footer = [None] * self.num_columns
        for column, prop in self._footers.items():
            try:
                index = self.headers.index(column)
            except ValueError:
                continue
            column = self.get_column(column)
            if prop['method'] == 'sum':
                value = sum(column)
            elif prop['method'] == 'max':
                value = max((r for r in column if r is not unknown))
            elif prop['method'] == 'mean':
                try:
                    weights = self.get_column(prop.get('weights'))
                    if isinstance(weights, str):
                        weights = self.get_column(weights)
                except KeyError:
                    weights = [1] * len(column)
                value = sum(
                    v * w for v, w in zip(column, weights)
                    if unknown not in (v, w))
                value /= sum(w for w in weights if w is not unknown)
                if any(isinstance(c, Percent) for c in column):
                    value = Percent(value)
            elif prop['method'] == 'value':
                value = prop['value']
            else:
                raise TypeError('Unrecognized method.')
            footer[index] = value
        return footer

    def _plumb_value(self, value):
        if value is None or value is unknown:
            return
        if isinstance(value, Percent):
            return float(value)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._plumb_value(v) for v in value]
        if isinstance(value, dict):
            return {
                self._plumb_value(k): self._plumb_value(v)
                for k, v in value.items()}
        return str(value)

    def plumb(self):
        infos = {'items': []}
        for row in self._rows:
            info = self._plumb_value({
                k: v for k, v in zip(self.headers, row)})
            infos['items'].append(info)
        if self._footers:
            footer = self._get_footers()
            infos['footer'] = self._plumb_value({
                key: value for key, value in zip(self.headers, footer)
                if value is not None})
        return infos

    def _footer_row(self, widths=None):
        if not self._footers:
            return None
        footer = self._get_footers()
        return self._format_row(footer, self._empty_formatters, widths=widths)

    def _column_widths(self):
        row_widths = []
        others = [self.headers, self._footer_row()]
        for row in self._rows + others:
            if row is None:
                continue
            if len(row) != self.num_columns:
                raise ValueError(
                    'Number of columns in row {} does not match headers {}.'
                    .format(row, self.headers))
            formatters = self._formatters
            if row in others:
                formatters = self._empty_formatters
            row = self._format_row(row, formatters)
            row_widths.append(len(e) for e in row)
        return [max(col) for col in zip(*row_widths)]

    def _format_rule(self, style, widths):
        left, col, mid, right = style
        content = col.join(
            mid * w for w, h in zip(widths, self.headers)
            if not h.endswith('_'))
        return f'{left}{content}{right}'

    def format(self):
        widths = self._column_widths()
        style = self.styles[self.style]
        # header
        header = []
        for i, h in enumerate(self.headers):
            header.append(self._format_value(h, None, widths[i]))
        table = [header]
        # rows
        for row in self._rows:
            table.append(self._format_row(row, self._formatters, widths))
        # footer
        footer = self._footer_row(widths)
        if footer:
            self.add_rule()
            table.append(footer)
        # lines
        lined_table = []
        column_left, column_mid, column_right = style['column']
        for row in table:
            row = column_mid.join(
                e for e, h in zip(row, self.headers)
                if not h.endswith('_'))
            lined_table.append(f'{column_left}{row}{column_right}')
        # rules
        ruled_table = []
        top = style.get('top')
        middle = style.get('middle')
        bottom = style.get('bottom')
        if top:
            ruled_table.append(self._format_rule(top, widths))
        for index, row in enumerate(lined_table):
            ruled_table.append(row)
            if middle and index in self._rules:
                ruled_table.append(self._format_rule(middle, widths))
        if bottom:
            ruled_table.append(self._format_rule(bottom, widths))
        return '\n'.join(ruled_table)

    def csv(self):
        rows = [', '.join(self.headers)]
        for r in self._rows:
            row = []
            for v in r:
                if isinstance(v, Percent):
                    v = float(v)
                row.append(str(v))
            rows.append(', '.join(row))
        return '\n'.join(rows)
